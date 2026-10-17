"""
stylized.py — does the simulated market look like a real one?

Prices in real markets are not a coin-flip random walk. Across assets,
countries and decades, returns share a handful of statistical regularities
that Rama Cont (2001, "Empirical properties of asset returns: stylized facts
and statistical issues") called STYLIZED FACTS. Any market model that claims
to be realistic should reproduce them; failing one tells you exactly which
mechanism the model is missing.

  1. Heavy tails       Big moves happen far more often than a bell curve
                       predicts (excess kurtosis > 0). Risk models that assume
                       normality underestimate crashes.
  2. No linear         Today's return does not predict tomorrow's (ACF ~ 0).
     autocorrelation   If it did, traders would arbitrage it away.
  3. Volatility        ...but the SIZE of today's move does predict tomorrow's:
     clustering        calm follows calm, storms follow storms. ACF of |r| is
                       positive and decays slowly.
  4. Aggregational     Sum returns over longer windows (week, month) and the
     Gaussianity       distribution looks more normal: kurtosis falls.
  5. Leverage effect   For equities, falling prices raise future volatility:
                       corr(r_t, |r_{t+k}|) < 0. Firms get more levered as
                       equity drops, and fear is asymmetric.

`facts(returns)` checks all five on any 1-D series. `compare()` runs them on
real daily returns and on the simulator's mid-price returns side by side.
`from_capture()` runs them on a Binance L2 capture file.

numpy/pandas only. Run `python -m backend.stylized` for the self-check.
"""

import math
import time

import numpy as np

try:
    from backend.data import get_prices
    from backend.market_sim import Simulation
    from engine.replay import best_pair, replay
except ImportError:                        # running this file directly
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend.data import get_prices
    from backend.market_sim import Simulation
    from engine.replay import best_pair, replay

MAX_LAG = 20
AGG = (1, 5, 20)
REAL_SOURCES = ("yahoo", "cache")   # data.get_prices only caches real downloads


def _kurt(x):
    """Excess kurtosis: 0 for a normal distribution, > 0 for fat tails."""
    x = np.asarray(x, float)
    d = x - x.mean()
    m2 = (d ** 2).mean()
    return float((d ** 4).mean() / m2 ** 2 - 3) if m2 > 0 else 0.0


def _skew(x):
    d = np.asarray(x, float) - np.mean(x)
    m2 = (d ** 2).mean()
    return float((d ** 3).mean() / m2 ** 1.5) if m2 > 0 else 0.0


def acf(x, max_lag=MAX_LAG):
    """Sample autocorrelation at lags 1..max_lag (standard biased estimator)."""
    d = np.asarray(x, float) - np.mean(x)
    var = (d ** 2).sum()
    if var == 0:
        return [0.0] * max_lag
    return [float((d[:-k] * d[k:]).sum() / var) for k in range(1, max_lag + 1)]


def hill(x, frac=0.05):
    """Hill estimator of the tail index alpha on |returns|.

    The tail of a power law decays like P(|r| > x) ~ x^-alpha. Real equity
    returns typically give alpha ~ 3 ("inverse cubic law"); a normal
    distribution has no power-law tail, so the estimate comes out large and
    keeps growing with sample size.
    """
    a = np.sort(np.abs(np.asarray(x, float)))[::-1]
    k = max(5, int(len(a) * frac))
    if k >= len(a) or a[k] <= 0:
        return None
    return float(1 / np.mean(np.log(a[:k] / a[k])))


def facts(returns, max_lag=MAX_LAG):
    """All five stylized facts for one return series, each with pass + number."""
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < max(100, 5 * max_lag):
        raise ValueError(f"need at least {max(100, 5 * max_lag)} returns, got {n}")
    if r.std() == 0:
        raise ValueError("returns are constant")
    band = 1.96 / math.sqrt(n)        # 95% band for the ACF of white noise

    # 1. Heavy tails. Even normal data shows some sample kurtosis, with
    # standard error sqrt(24/n), so demand 3 standard errors, not just > 0.
    k = _kurt(r)
    k_se = math.sqrt(24 / n)
    alpha = hill(r)

    # 2. Linear ACF. Lag 1 is reported but excused: bid-ask bounce and
    # stale prices legitimately create a small lag-1 effect in real data.
    # Under the null ~5% of lags poke out of the band by chance, so allow 2/19.
    # Caveat: the +/-1.96/sqrt(n) band assumes i.i.d. returns. With volatility
    # clustering the true sampling error of the ACF is wider, so real data can
    # "fail" this test a little more often than the band suggests.
    ac = acf(r, max_lag)
    outside = sum(abs(v) > band for v in ac[1:])

    # 3. Volatility clustering: ACF of |r| above the band at most lags.
    ac_abs = acf(np.abs(r), max_lag)
    above = sum(v > band for v in ac_abs)

    # 4. Aggregational Gaussianity: non-overlapping sums over m periods.
    # Only meaningful if there were heavy tails to lose in the first place.
    # Pass = monthly-scale (20) kurtosis below the 1-period one. The 5-period
    # value is reported but not required: sample kurtosis of fat-tailed data
    # is itself very noisy, and with persistent volatility the decay is slow.
    agg = {m: _kurt(r[: n // m * m].reshape(-1, m).sum(1)) for m in AGG}

    # 5. Leverage: corr(r_t, |r_{t+k}|). Negative = drops precede volatility.
    def lev(kk):
        return float(np.corrcoef(r[:-kk], np.abs(r[kk:]))[0, 1])
    leverage = [lev(kk) for kk in range(1, 11)]
    lev_mean = float(np.mean(leverage[:5]))

    heavy = k > 3 * k_se
    return {
        "n": n,
        "mean": float(r.mean()),
        "std": float(r.std()),
        "skew": _skew(r),
        "zero_fraction": float(np.mean(r == 0)),
        "band": band,
        "acf": ac,
        "acf_abs": ac_abs,
        "leverage_corr": leverage,
        "heavy_tails": {"pass": bool(heavy), "excess_kurtosis": k,
                        "threshold": 3 * k_se, "hill_alpha": alpha},
        "no_autocorrelation": {"pass": bool(outside <= 2), "lag1": ac[0],
                               "lags_outside_band_2_to_20": int(outside)},
        "volatility_clustering": {"pass": bool(above >= max_lag // 2),
                                  "lags_above_band": int(above),
                                  "abs_acf_lag1": ac_abs[0],
                                  "abs_acf_lag20": ac_abs[-1]},
        "aggregational_gaussianity": {
            "pass": bool(heavy and agg[20] < agg[1]),
            "kurtosis_by_horizon": {str(m): v for m, v in agg.items()}},
        "leverage_effect": {"pass": bool(lev_mean < -band),
                            "mean_corr_lags_1_5": lev_mean},
    }


FACT_NAMES = ("heavy_tails", "no_autocorrelation", "volatility_clustering",
              "aggregational_gaussianity", "leverage_effect")


def histogram(returns, lo=-5.0, hi=5.0, width=0.25):
    """Density of standardized returns vs the standard normal, for a chart."""
    r = np.asarray(returns, float)
    z = (r - r.mean()) / r.std()
    edges = np.arange(lo, hi + width / 2, width)
    counts, _ = np.histogram(z, edges)
    centers = (edges[:-1] + edges[1:]) / 2
    return {
        "centers": [round(float(c), 3) for c in centers],
        "density": [float(c / (len(z) * width)) for c in counts],
        "normal": [float(math.exp(-c * c / 2) / math.sqrt(2 * math.pi)) for c in centers],
        "outside_fraction": float(np.mean(np.abs(z) > hi)),
    }


def _findings(real, sim, source):
    out = []
    if source not in REAL_SOURCES:
        out.append(f"Data source is '{source}': SYNTHETIC offline prices (i.i.d. normal "
                   "returns), not real market data. The 'real' column is a placebo here; "
                   "rerun with internet access for genuine results.")
    other = "the market data" if source in REAL_SOURCES else "the synthetic stand-in"
    for f in FACT_NAMES:
        label = f.replace("_", " ")
        if real[f]["pass"] and not sim[f]["pass"]:
            out.append(f"The simulator does NOT reproduce {label}; {other} does.")
        elif sim[f]["pass"] and not real[f]["pass"]:
            out.append(f"The simulator shows {label} but {other} does not.")
        elif sim[f]["pass"]:
            out.append(f"Both show {label}.")
        else:
            out.append(f"Neither shows {label}.")
    if sim["zero_fraction"] > 0.3:
        out.append(f"{sim['zero_fraction']:.0%} of simulator steps have an unchanged mid. "
                   "Its kurtosis is partly tick discreteness (many zeros plus occasional "
                   "one-tick jumps), not a power-law tail; compare the Hill alpha.")
    if sim["acf"][0] > 3 * sim["band"]:
        out.append(f"Simulator returns trend: lag-1 ACF {sim['acf'][0]:+.2f} (band "
                   f"+/-{sim['band']:.3f}). Real prices are close to unpredictable; here "
                   "momentum agents chase moves and the mid tends to walk one tick at a "
                   "time, so a move is likely to be followed by another in the same direction. "
                   "Nobody in the simulator is arbitraging that predictability away.")
    if not sim["leverage_effect"]["pass"]:
        out.append("No leverage effect in the simulator is expected: its agents are "
                   "symmetric in up and down moves, and nothing in it plays the role of "
                   "a firm whose leverage rises as its stock falls.")
    return out


def compare(ticker="SPY", start="2015-01-01", end="2025-01-01", sim_steps=3000, seed=7):
    """Stylized facts: real daily log returns vs simulator per-step mid returns."""
    df, source = get_prices(ticker, start, end)
    close = df["Close"].astype(float).to_numpy()
    real_r = np.diff(np.log(close))

    sim = Simulation(seed=seed)
    sim.run(int(sim_steps))
    sim_r = np.diff(np.log(np.asarray(sim.history, float)))

    real, simf = facts(real_r), facts(sim_r)
    pick = lambda fx: {f: fx[f] for f in FACT_NAMES}
    return {
        "ticker": ticker.upper(), "start": start, "end": end,
        "data_source": source,
        "data_is_real": source in REAL_SOURCES,
        "data_note": ("real market data" if source in REAL_SOURCES else
                      "'sample' = synthetic offline fallback (normal random walk); "
                      "the 'real' facts are only real when data comes from Yahoo"),
        "sim_steps": int(sim_steps), "seed": seed,
        "lags": list(range(1, MAX_LAG + 1)),
        "real": {"facts": pick(real), "n": real["n"], "std": real["std"],
                 "skew": real["skew"], "band": real["band"], "acf": real["acf"],
                 "acf_abs": real["acf_abs"], "leverage_corr": real["leverage_corr"],
                 "histogram": histogram(real_r)},
        "sim": {"facts": pick(simf), "n": simf["n"], "std": simf["std"],
                "skew": simf["skew"], "zero_fraction": simf["zero_fraction"],
                "band": simf["band"], "acf": simf["acf"], "acf_abs": simf["acf_abs"],
                "leverage_corr": simf["leverage_corr"], "histogram": histogram(sim_r)},
        "findings": _findings(real, simf, source),
    }


def from_capture(path):
    """Stylized facts on mid-price returns from a Binance L2 capture file.

    Event time, not clock time: one return per applied book update where the
    mid changed. Updates that leave the mid unchanged are skipped, otherwise
    a quiet book would flood the series with zeros.
    """
    mids, trades = [], 0
    for kind, _msg, book in replay(path):
        if kind == "trade":
            trades += 1
            continue
        pair = best_pair(book)
        if pair is None:
            continue
        mid = (pair[0][0] + pair[1][0]) / 2
        if not mids or mid != mids[-1]:
            mids.append(mid)
    r = np.diff(np.log(np.asarray(mids, float)))
    out = facts(r)
    out.update({"path": str(path), "mid_changes": len(r), "trades": trades,
                "histogram": histogram(r)})
    return out


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
def _garch(n, seed=0, omega=1e-6, a=0.1, b=0.85, df=5):
    """GARCH(1,1) with Student-t shocks: today's variance feeds on yesterday's
    squared shock. That feedback is exactly what produces clustering and fat
    tails; the t shocks make the tails as fat as real daily returns."""
    rng = np.random.default_rng(seed)
    r = np.empty(n)
    var = omega / (1 - a - b)
    scale = math.sqrt(df / (df - 2))          # unit-variance t
    for t in range(n):
        r[t] = math.sqrt(var) * rng.standard_t(df) / scale
        var = omega + a * r[t] ** 2 + b * var
    return r


def demo():
    import gzip
    import json
    import tempfile

    # positive control: GARCH has heavy tails and clustering, no linear ACF
    g = facts(_garch(5000))
    assert g["heavy_tails"]["pass"], g["heavy_tails"]
    assert g["volatility_clustering"]["pass"], g["volatility_clustering"]
    assert g["no_autocorrelation"]["pass"], g["no_autocorrelation"]
    assert g["aggregational_gaussianity"]["pass"], g["aggregational_gaussianity"]

    # negative control: i.i.d. normal noise must FAIL the tail/clustering tests,
    # otherwise the tests detect nothing. Several seeds, not one lucky one.
    for seed in range(5):
        z = facts(np.random.default_rng(100 + seed).standard_normal(5000))
        assert not z["heavy_tails"]["pass"], ("false positive tails", seed, z["heavy_tails"])
        assert not z["volatility_clustering"]["pass"], ("false clustering", seed)
        assert not z["leverage_effect"]["pass"], ("false leverage", seed)
        assert z["no_autocorrelation"]["pass"], ("white noise flagged as autocorrelated", seed)

    # a series with built-in leverage (drops raise next-period vol) must pass it
    rng = np.random.default_rng(1)
    r = np.empty(5000)
    vol = 0.01
    for t in range(5000):
        r[t] = vol * rng.standard_normal()
        vol = 0.01 * (1 + 3 * max(0.0, -r[t]) / 0.01) ** 0.5
    assert facts(r)["leverage_effect"]["pass"], "leverage series not detected"

    try:
        facts([0.01, -0.01] * 10)
        raise AssertionError("short series must be rejected")
    except ValueError:
        pass

    # compare() end to end
    t0 = time.perf_counter()
    c = compare()
    t_cmp = time.perf_counter() - t0
    assert t_cmp < 20, f"compare too slow: {t_cmp:.1f}s"
    assert c["data_source"] in ("yahoo", "cache", "sample")
    assert len(c["real"]["acf"]) == len(c["sim"]["acf"]) == MAX_LAG
    h = c["sim"]["histogram"]
    assert len(h["centers"]) == len(h["density"]) == len(h["normal"])
    assert c["findings"]
    json.dumps(c)                                   # JSON-ready
    assert compare(sim_steps=500) == compare(sim_steps=500), "must be deterministic"

    # from_capture on a tiny synthetic Binance capture: a random-walk mid
    with tempfile.TemporaryDirectory() as d:
        p = f"{d}/cap.jsonl.gz"
        rets = _garch(400, seed=3, omega=1e-8)
        mid = 68000 * np.exp(np.cumsum(rets))
        bid, ask = round(mid[0] - 0.5, 2), round(mid[0] + 0.5, 2)
        rows = [{"type": "snapshot", "lastUpdateId": 100,
                 "bids": [[f"{bid:.2f}", "1.0"]], "asks": [[f"{ask:.2f}", "1.0"]]}]
        uid = 100
        for i, m in enumerate(mid[1:]):
            nb, na = round(m - 0.5, 2), round(m + 0.5, 2)
            rows.append({"type": "diff", "U": uid + 1, "u": uid + 1,
                         "b": [[f"{bid:.2f}", "0"], [f"{nb:.2f}", "1.0"]],
                         "a": [[f"{ask:.2f}", "0"], [f"{na:.2f}", "1.0"]]})
            uid += 1
            bid, ask = nb, na
            if i % 10 == 0:
                rows.append({"type": "trade", "p": f"{bid:.2f}", "q": "0.1",
                             "m": True, "T": i})
        with gzip.open(p, "wt") as fh:
            fh.write("\n".join(json.dumps(x) for x in rows))
        cap = from_capture(p)
        assert cap["mid_changes"] > 300, cap["mid_changes"]
        assert cap["trades"] == 40
        assert cap["heavy_tails"]["excess_kurtosis"] > 0

    print(f"ok — GARCH passes tails/clustering, Gaussian noise fails them (5 seeds), "
          f"leverage detected, from_capture {cap['mid_changes']} mid changes")
    print(f"ok — compare() in {t_cmp:.2f}s, data_source={c['data_source']}")
    for side in ("real", "sim"):
        fx = c[side]["facts"]
        print(f"   {side:4s}: " + ", ".join(f"{f}={'PASS' if fx[f]['pass'] else 'fail'}"
                                         for f in FACT_NAMES))
    for line in c["findings"]:
        print("   -", line)


if __name__ == "__main__":
    demo()
