"""
portfolio.py — multi-asset backtests, position sizing and risk numbers.

backtest.py answers "what if I traded ONE stock, all-in or all-out?". Real
money is managed as a PORTFOLIO: several assets held at once, each with a
weight (the fraction of the account it gets). Two questions matter:

  1. SIZING — how much of each? The schemes here:
       equal        1/N each. Naive, but famously hard to beat.
       inverse_vol  weight proportional to 1/volatility, so a calm asset
                    (bonds) gets more money than a wild one (tech stocks).
                    Each asset then contributes roughly the same *swing* to
                    the book instead of the riskiest asset dominating it.
       vol_target   inverse-vol weights, then the whole book is scaled up or
                    down so its *expected* volatility hits a target (e.g.
                    10%/yr). In calm markets you hold more, in stormy markets
                    less, keeping risk roughly constant over time.
       risk_parity  "equal risk contribution": like inverse-vol, but it also
                    accounts for correlations, so two assets that move
                    together are treated as partly the same bet.

  2. RISK — what could go wrong? VaR/CVaR (how bad is a bad day), beta (how
     much of this is just "the market"), factor exposures (which well-known
     return drivers is this portfolio secretly riding), correlations and
     risk contributions (who is actually driving the swings).

No lookahead: weights are decided with data up to the close of day T and
held from day T+1, the same rule backtest.py applies to signals.
"""

import numpy as np
import pandas as pd

from backend.backtest import TRADING_DAYS_PER_YEAR, run_backtest, summary_metrics
from backend.data import get_prices
from backend.strategies import STRATEGIES, clean_params

WEIGHTINGS = ("equal", "inverse_vol", "vol_target", "risk_parity")
REBALANCES = ("monthly", "weekly")
MIN_TICKERS, MAX_TICKERS = 2, 10
MAX_WEIGHT_POINTS = 300        # downsample the weights-over-time table to this

# Free ETF proxies for the academic (Fama-French / Carhart) factors. The real
# factors are long-short portfolios built from thousands of stocks; these ETF
# spreads are rough stand-ins, chosen only because they are free on Yahoo.
#   market   = SPY            (S&P 500)
#   size     = IWM - SPY      (small caps minus large caps, "SMB")
#   value    = IWD - IWF      (Russell 1000 value minus growth, "HML")
#   momentum = MTUM - SPY     (momentum ETF minus market; MTUM starts 2013-04)
FACTOR_TICKERS = ("SPY", "IWM", "IWD", "IWF", "MTUM")

# Standard normal quantiles / density, so we don't need scipy.
_Z = {0.95: 1.6448536269514722, 0.99: 2.3263478740408408}


def _phi(z):
    return np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)


def _metrics(ret: np.ndarray, starting_cash: float) -> tuple[dict, np.ndarray]:
    """backtest.summary_metrics (the project's one metric definition) + equity."""
    metrics, _ = summary_metrics(ret, starting_cash)
    return metrics, starting_cash * np.cumprod(1 + ret)


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------
def _target_weights(window: np.ndarray, weighting: str, target_vol: float):
    """Weights from a (lookback x K) window of past daily returns ONLY.

    Returns (weights, annualized covariance matrix of the window).
    """
    k = window.shape[1]
    cov = np.cov(window, rowvar=False) * TRADING_DAYS_PER_YEAR
    if weighting == "equal":
        return np.full(k, 1.0 / k), cov

    # Inverse volatility: an asset twice as volatile gets half the money, so
    # each one moves the account by about the same amount on a typical day.
    sd = np.maximum(np.sqrt(np.diag(cov)), 1e-12)
    w = (1 / sd) / (1 / sd).sum()

    if weighting == "risk_parity":
        # Equal risk contribution: we want w_i * (C w)_i equal for every i,
        # where (C w)_i is how much asset i co-moves with the whole book.
        # Fixed-point iteration w_i <- 1 / (C w)_i, damped with a square root
        # so it doesn't oscillate, starting from inverse-vol. Converges fast
        # for the long-only, positive-definite case we have here.
        for _ in range(200):
            w = np.sqrt(w / np.maximum(cov @ w, 1e-18))
            w /= w.sum()
    elif weighting == "vol_target":
        # Ex-ante (forward-looking) portfolio vol = sqrt(w' C w), using the
        # covariance matrix, so correlations count: diversification lowers it.
        # Scale the book so that number hits target_vol. If the book is
        # already too CALM we would need to lever up past 100% of equity;
        # we cap gross exposure at 1.0 instead: this account never borrows,
        # the leftover just sits in cash.
        pvol = float(np.sqrt(w @ cov @ w))
        if pvol > 0:
            w = w * min(target_vol / pvol, 1.0)
    return w, cov


def _rebalance_mask(dates: pd.DatetimeIndex, rebalance: str) -> np.ndarray:
    """True on the LAST trading day of each month/week: decide at that close."""
    if rebalance == "never":
        return np.zeros(len(dates), dtype=bool)
    period = dates.to_period("M" if rebalance == "monthly" else "W")
    # Day i is a period end if day i+1 falls in a different month/week.
    return np.r_[np.asarray(period[:-1] != period[1:]), True]


def _simulate(R, sig, mask, weighting, target_vol, lookback, commission):
    """Day-by-day book. R: (N x K) daily returns, sig: (N x K) 0/1 gates.

    Nothing is invested until `lookback` returns exist, for EVERY weighting,
    so all schemes (and the benchmark) are compared over the same window.
    Returns (net daily returns from day s, held weights from day s,
    turnover per rebalance, (weights, cov) at the last rebalance).
    """
    n, k = R.shape
    s = lookback
    w = np.zeros(k)                # weights held (fraction of equity); rest is cash
    base, cov = np.zeros(k), None
    net = np.zeros(n)
    held = np.zeros((n, k))
    pending, turnovers, last = 0.0, [], (np.zeros(k), np.eye(k))
    for t in range(s, n):
        if t > s:
            # Day t's return is earned on weights decided at the close of t-1.
            g = float(w @ R[t])
            net[t] = g - pending   # commission for yesterday's trades, charged
            pending = 0.0          # on the first day held, like backtest.py
            # Between rebalances weights DRIFT: a winner's share of the book
            # grows. Cash (1 - sum w) earns 0 here.
            w = w * (1 + R[t]) / (1 + g) if 1 + g > 0 else w
        rebal = t == s or mask[t]
        if rebal or (sig[t] != sig[t - 1]).any():
            if rebal:   # sizing uses ONLY returns up to and including day t
                base, cov = _target_weights(R[t - lookback + 1:t + 1], weighting, target_vol)
            # A strategy signal of 0 sends that sleeve to cash (its weight is
            # NOT handed to the others), so gating only ever lowers exposure.
            target = base * sig[t]
            to = float(np.abs(target - w).sum())   # fraction of equity traded
            pending = to * commission
            turnovers.append(to)
            w = target
            last = (target, cov)
        held[t] = w
    return net[s:], held[s:], turnovers, last


# ---------------------------------------------------------------------------
# Risk measures
# ---------------------------------------------------------------------------
def _var_cvar(ret: np.ndarray) -> dict:
    """1-day Value at Risk and Conditional VaR, as positive % of the book.

    VaR 95%: "on 95% of days you lose less than this" (a threshold).
    CVaR 95% (expected shortfall): "on the worst 5% of days, you lose THIS
    much on average". CVaR looks past the threshold into the tail, so it is
    always >= VaR, and regulators increasingly prefer it.

    Historical = read straight off the actual return distribution.
    Parametric = assume returns are Gaussian with the sample mean and stdev.
    Real returns have FAT TAILS (crashes are far more common than a bell
    curve says), so at extreme levels (99%) historical VaR/CVaR usually come
    out larger than parametric: the Gaussian under-states crash risk. At 95%
    the two are often close, and the Gaussian can even be slightly higher,
    because a fat-tailed distribution with the same stdev is also more
    peaked in the middle.
    """
    mu, sd = ret.mean(), ret.std(ddof=1)
    out = {}
    for lvl in (0.95, 0.99):
        tag = int(lvl * 100)
        q = np.quantile(ret, 1 - lvl)
        out[f"var{tag}_hist_pct"] = round(float(-q) * 100, 3)
        out[f"cvar{tag}_hist_pct"] = round(float(-ret[ret <= q].mean()) * 100, 3)
        z = _Z[lvl]
        out[f"var{tag}_param_pct"] = round(float(-(mu - z * sd)) * 100, 3)
        out[f"cvar{tag}_param_pct"] = round(float(-(mu - sd * _phi(z) / (1 - lvl))) * 100, 3)
    return out


def _ols(y: np.ndarray, X: np.ndarray):
    """Ordinary least squares with an intercept. Returns (coef, t-stats, R^2).

    coef[0] is the intercept (daily alpha), coef[1:] the betas. Standard
    errors are the classic homoskedastic ones: sqrt(diag(s^2 (X'X)^-1)).
    """
    X1 = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ coef
    dof = max(len(y) - X1.shape[1], 1)
    s2 = resid @ resid / dof
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(X1.T @ X1)))
    tstat = np.divide(coef, se, out=np.zeros_like(coef), where=se > 0)
    sst = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid @ resid) / sst if sst > 0 else 0.0
    return coef, tstat, float(r2)


def _risk_contrib(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Share of portfolio vol coming from each asset: w_i (C w)_i / sigma_p^2.

    (C w)_i / sigma_p is asset i's MARGINAL contribution (how much book vol
    rises per extra unit of i); times w_i gives its contribution, and the
    contributions add up to sigma_p. Dividing by sigma_p again gives shares
    that sum to 1. Inverse-vol with correlated assets is NOT equal here;
    risk parity is.
    """
    pvar = float(w @ cov @ w)
    return w * (cov @ w) / pvar if pvar > 0 else np.zeros_like(w)


# ---------------------------------------------------------------------------
# Core (pure: takes prices, no downloads) — demo() tests this directly
# ---------------------------------------------------------------------------
def _core(close: pd.DataFrame, signals: pd.DataFrame | None, weighting="equal",
          target_vol=0.10, lookback=60, rebalance="monthly",
          starting_cash=10_000.0, commission_pct=0.001) -> dict:
    tickers = list(close.columns)
    R = close.pct_change().fillna(0).to_numpy()
    sig = np.ones_like(R) if signals is None else signals.reindex(
        close.index).fillna(0).clip(0, 1).to_numpy(dtype=float)
    mask = _rebalance_mask(close.index, rebalance)

    net, held, turns, (w_last, cov_last) = _simulate(
        R, sig, mask, weighting, target_vol, lookback, commission_pct)
    metrics, equity = _metrics(net, starting_cash)
    years = len(net) / TRADING_DAYS_PER_YEAR
    metrics["avg_turnover_pct"] = round(float(np.mean(turns)) * 100, 2)
    metrics["annual_turnover_pct"] = round(float(np.sum(turns) / years) * 100, 1)
    metrics["num_rebalances"] = len(turns)
    metrics["avg_gross_exposure_pct"] = round(float(held.sum(1).mean()) * 100, 1)

    # Benchmark: equal-weight buy-and-hold, bought on the same day, never
    # rebalanced (weights drift), entry commission only.
    b_net, *_ = _simulate(R, np.ones_like(R), _rebalance_mask(close.index, "never"),
                          "equal", target_vol, lookback, commission_pct)
    b_metrics, b_equity = _metrics(b_net, starting_cash)

    live = net[1:]   # first day is the decision day, return 0 by construction
    rc = _risk_contrib(w_last, cov_last)
    pvol = float(np.sqrt(w_last @ cov_last @ w_last))
    corr = pd.DataFrame(R[lookback + 1:], columns=tickers).corr().round(3)
    return {
        "dates": close.index[lookback:],
        "net": net,
        "equity": equity,
        "weights": held,
        "metrics": metrics,
        "benchmark": {"equity": b_equity, "metrics": b_metrics},
        "risk": {
            **_var_cvar(live),
            "correlation": {"tickers": tickers, "matrix": corr.fillna(0).values.tolist()},
            "risk_contribution": {
                "weights": {t: round(float(x), 4) for t, x in zip(tickers, w_last)},
                "share_of_vol_pct": {t: round(float(x) * 100, 2) for t, x in zip(tickers, rc)},
                "ex_ante_vol_pct": round(pvol * 100, 2),
            },
        },
    }


def _factor_block(port: pd.Series, start: str, end: str) -> tuple[dict, dict]:
    """Beta to SPY plus a 4-factor OLS on free ETF proxies."""
    closes, sources = {}, {}
    for f in FACTOR_TICKERS:
        try:
            df, src = get_prices(f, start, end)
            closes[f], sources[f] = df["Close"], src
        except Exception:
            pass
    if "SPY" not in closes:
        return {"error": "SPY unavailable"}, sources
    # pct_change per series BEFORE joining, so a late starter (MTUM) is just
    # NaN early on instead of forcing everything else to start late.
    rets = pd.DataFrame({f: c.pct_change() for f, c in closes.items()})
    mkt = rets["SPY"]

    # Beta: how many % the portfolio moves for a 1% market move.
    # Correlation: how tightly it follows the market (1 = lockstep).
    both = pd.concat([port, mkt], axis=1, join="inner").dropna()
    y, x = both.iloc[:, 0].to_numpy(), both.iloc[:, 1].to_numpy()
    out = {
        "beta_spy": round(float(np.cov(y, x)[0, 1] / x.var(ddof=1)), 3) if len(x) > 2 else None,
        "corr_spy": round(float(np.corrcoef(y, x)[0, 1]), 3) if len(x) > 2 else None,
    }

    # Factor regression. Each beta answers "how much of this style is the
    # portfolio carrying?" e.g. size beta 0.3 = it behaves partly like a
    # small-cap tilt. Alpha is what's left after all factors: the part of
    # the return these well-known (and cheaply replicable) bets can't explain.
    # Excess returns: there's no free daily risk-free series here, so rf = 0,
    # the same convention as backtest.py's Sharpe. Over short daily horizons
    # this barely moves the betas; it does inflate alpha by roughly rf/yr.
    facs = {"market": mkt}
    if {"IWM"} <= closes.keys():
        facs["size"] = rets["IWM"] - mkt
    if {"IWD", "IWF"} <= closes.keys():
        facs["value"] = rets["IWD"] - rets["IWF"]
    if "MTUM" in closes:
        facs["momentum"] = rets["MTUM"] - mkt
    data = pd.concat([port.rename("port"), pd.DataFrame(facs)], axis=1, join="inner").dropna()
    if len(data) < 60:
        out["factors"] = {"error": "fewer than 60 overlapping days with factor data"}
        return out, sources
    coef, t, r2 = _ols(data["port"].to_numpy(), data[list(facs)].to_numpy())
    out["factors"] = {
        "betas": {f: round(float(b), 3) for f, b in zip(facs, coef[1:])},
        "t_stats": {f: round(float(v), 2) for f, v in zip(facs, t[1:])},
        "alpha_annual_pct": round(float(coef[0]) * TRADING_DAYS_PER_YEAR * 100, 2),
        "alpha_t_stat": round(float(t[0]), 2),
        "r_squared": round(r2, 3),
        # The regression only uses days where every factor exists (MTUM
        # trims anything before 2013-04 with real data); report the window.
        "window": {"start": str(data.index[0].date()), "end": str(data.index[-1].date()),
                   "n_obs": len(data)},
        "proxies": "market=SPY, size=IWM-SPY, value=IWD-IWF, momentum=MTUM-SPY "
                   "(rough free stand-ins for Fama-French/Carhart factors)",
    }
    return out, sources


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run(tickers, start, end, strategy=None, params=None, weighting="equal",
        target_vol=0.10, lookback=60, rebalance="monthly",
        starting_cash=10_000.0, commission_pct=0.001) -> dict:
    """Multi-asset backtest. Raises ValueError on bad input (-> HTTP 400)."""
    tickers = list(dict.fromkeys(t.upper().strip() for t in tickers if str(t).strip()))
    if not MIN_TICKERS <= len(tickers) <= MAX_TICKERS:
        raise ValueError(f"Provide {MIN_TICKERS} to {MAX_TICKERS} distinct tickers.")
    if weighting not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {', '.join(WEIGHTINGS)}")
    if rebalance not in REBALANCES:
        raise ValueError(f"rebalance must be one of {', '.join(REBALANCES)}")
    lookback = int(lookback)
    if not 20 <= lookback <= 252:
        raise ValueError("lookback must be between 20 and 252 days.")
    if not 0.01 <= float(target_vol) <= 1.0:
        raise ValueError("target_vol must be between 0.01 and 1.0 (annualized).")
    if not starting_cash or starting_cash <= 0:
        raise ValueError("starting_cash must be greater than zero.")
    if commission_pct < 0:
        raise ValueError("commission_pct cannot be negative.")
    clean = clean_params(strategy, params) if strategy else {}

    frames, sources = {}, {}
    for t in tickers:
        try:
            frames[t], sources[t] = get_prices(t, start, end)
        except Exception as exc:
            raise ValueError(f"{t}: {exc}")
    # Inner join: only dates where every ticker trades (see /api/compare).
    close = pd.concat({t: f["Close"] for t, f in frames.items()}, axis=1, join="inner").dropna()
    if len(close) < lookback + 30:
        raise ValueError(f"Need at least {lookback + 30} common trading days; got {len(close)}.")

    signals = None
    if strategy:
        func = STRATEGIES[strategy]["func"]
        signals = pd.DataFrame({t: func(frames[t].reindex(close.index), **clean)
                                for t in tickers}, index=close.index)

    core = _core(close, signals, weighting, float(target_vol), lookback, rebalance,
                 float(starting_cash), float(commission_pct))
    dates = core["dates"]
    port = pd.Series(core["net"], index=dates).iloc[1:]
    fac, fac_sources = _factor_block(port, start, end)
    core["risk"].update(fac)
    core["risk"]["factor_data_sources"] = fac_sources

    step = max(1, -(-len(dates) // MAX_WEIGHT_POINTS))
    idx = list(range(0, len(dates), step))
    if idx[-1] != len(dates) - 1:
        idx.append(len(dates) - 1)
    W = core["weights"]

    notes = [
        "Survivorship bias: these tickers were picked today, with hindsight. "
        "Assets that went bust or were delisted aren't on anyone's list, so "
        "backtests of today's survivors look better than reality was.",
        "Weights are decided at the close of day T using data up to T only, "
        "and held from T+1. Between rebalances weights drift with prices.",
        f"Nothing is invested for the first {lookback} days (the lookback "
        "window), for every weighting and the benchmark, so they compare fairly.",
        "Long-only, no leverage: weights sum to at most 100%; the rest is cash earning 0%.",
        "Sharpe/Sortino use a 0% risk-free rate, the same as single-ticker backtests.",
    ]
    if weighting == "vol_target":
        notes.append("vol_target caps gross exposure at 100%: when markets are calm "
                     "the book can't reach the target without borrowing, so it doesn't.")
    if any(s == "sample" for s in {**sources, **fac_sources}.values()):
        notes.append("SAMPLE DATA: some series are synthetic random walks (offline "
                     "fallback). They are independent with similar volatility, so "
                     "weights, correlations, beta and factor numbers are not meaningful.")

    return {
        "tickers": tickers,
        "weighting": weighting,
        "rebalance": rebalance,
        "strategy": strategy,
        "dates": [str(d.date()) for d in dates],
        "equity": [round(float(v), 2) for v in core["equity"]],
        "weights": {
            "dates": [str(dates[i].date()) for i in idx],
            "by_ticker": {t: [round(float(W[i, j]), 4) for i in idx]
                          for j, t in enumerate(tickers)},
            "cash": [round(float(1 - W[i].sum()), 4) for i in idx],
        },
        "metrics": core["metrics"],
        "benchmark": {
            "label": "Equal-weight buy & hold",
            "equity": [round(float(v), 2) for v in core["benchmark"]["equity"]],
            "metrics": core["benchmark"]["metrics"],
        },
        "risk": core["risk"],
        "data_sources": sources,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Self-check: synthetic, offline, deterministic
# ---------------------------------------------------------------------------
def _synthetic(n, vols, corr=0.0, seed=0, df=None, start="2010-01-01"):
    """Correlated daily returns with given annual vols -> price DataFrame.

    df=None gives Gaussian returns; df=3 gives fat-tailed Student-t returns
    rescaled to the same stdev.
    """
    rng = np.random.default_rng(seed)
    k = len(vols)
    C = np.full((k, k), corr) + (1 - corr) * np.eye(k)
    L = np.linalg.cholesky(C)
    z = rng.standard_normal((n, k)) if df is None else \
        rng.standard_t(df, (n, k)) / np.sqrt(df / (df - 2))
    r = (z @ L.T) * (np.asarray(vols) / np.sqrt(TRADING_DAYS_PER_YEAR)) + 0.0002
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=idx,
                        columns=[f"A{i}" for i in range(k)])


def demo():
    # 1. Two identical series, equal weight, no costs == the single series,
    #    and our metrics match backtest.py's on the same path.
    one = _synthetic(800, [0.2], seed=1)
    two = pd.concat([one["A0"].rename("X"), one["A0"].rename("Y")], axis=1)
    lb = 60
    res = _core(two, None, "equal", lookback=lb, commission_pct=0.0)
    bt = run_backtest(one.iloc[lb:].rename(columns={"A0": "Close"}),
                      pd.Series(1, index=one.index[lb:]), commission_pct=0.0)
    assert np.allclose(res["equity"], bt["buy_hold"], atol=0.01), "2 identical != 1"
    for key in ("total_return_pct", "cagr_pct", "max_drawdown_pct", "calmar_ratio"):
        assert abs(res["metrics"][key] - bt["metrics"][key]) <= 0.011, key
    print(f"  1. equal-weight twin == single series, final {res['metrics']['final_equity']}")

    # 2. Inverse-vol overweights the calmer asset.
    px = _synthetic(1000, [0.08, 0.30], corr=0.2, seed=2)
    res = _core(px, None, "inverse_vol")
    w = res["weights"].mean(0)
    assert w[0] > 2 * w[1], f"inverse-vol weights {w}"
    print(f"  2. inverse-vol avg weights low-vol {w[0]:.2f} vs high-vol {w[1]:.2f}")

    # 3. Vol targeting hits the target on stationary data; never levers.
    px = _synthetic(2500, [0.10, 0.20, 0.30], corr=0.3, seed=3)
    res = _core(px, None, "vol_target", target_vol=0.08)
    rv = res["metrics"]["annual_vol_pct"]
    assert abs(rv - 8.0) < 1.0, f"realized vol {rv}% vs target 8%"
    assert res["weights"].sum(1).max() <= 1 + 1e-9
    hi = _core(px, None, "vol_target", target_vol=0.60)   # unreachable target
    assert hi["weights"].sum(1).max() <= 1 + 1e-9, "vol_target levered up"
    print(f"  3. vol_target 8% -> realized {rv}%, max gross "
          f"{hi['weights'].sum(1).max():.3f} with a 60% target")

    # 4. No lookahead: rewriting prices AFTER date D changes nothing up to D.
    px = _synthetic(600, [0.1, 0.2, 0.3], corr=0.2, seed=4)
    d = 400
    bumped = px.copy()
    bumped.iloc[d + 1:] *= np.linspace(1, 3, len(px) - d - 1)[:, None] ** [1, -1, 2]
    for wt in WEIGHTINGS:
        a = _core(px, None, wt)["weights"]
        b = _core(bumped, None, wt)["weights"]
        k = d - 60 + 1                       # weights rows start at lookback
        assert np.allclose(a[:k], b[:k]), f"lookahead in {wt}"
        assert not np.allclose(a, b), "perturbation had no effect at all?"
    print("  4. no lookahead: weights up to D unchanged when the future changes")

    # 5. VaR/CVaR ordering, and fat tails beat the Gaussian at 99%.
    fat = _synthetic(20000, [0.2], seed=5, df=3)["A0"].pct_change().dropna().to_numpy()
    v = _var_cvar(fat)
    for kind in ("hist", "param"):
        assert v[f"var99_{kind}_pct"] >= v[f"var95_{kind}_pct"]
        for lvl in (95, 99):
            assert v[f"cvar{lvl}_{kind}_pct"] >= v[f"var{lvl}_{kind}_pct"]
    assert v["var99_hist_pct"] > v["var99_param_pct"], v
    assert v["cvar99_hist_pct"] > v["cvar99_param_pct"], v
    print(f"  5. t(3) VaR99 hist {v['var99_hist_pct']}% > param {v['var99_param_pct']}%; "
          f"VaR95 hist {v['var95_hist_pct']}% vs param {v['var95_param_pct']}%")

    # 6. OLS recovers known betas.
    rng = np.random.default_rng(6)
    mkt, smb = rng.normal(0, 0.01, 3000), rng.normal(0, 0.005, 3000)
    y = 0.0001 + 1.2 * mkt + 0.5 * smb + rng.normal(0, 0.003, 3000)
    coef, t, r2 = _ols(y, np.column_stack([mkt, smb]))
    assert abs(coef[1] - 1.2) < 0.03 and abs(coef[2] - 0.5) < 0.05, coef
    assert t[1] > 50 and r2 > 0.9
    print(f"  6. OLS betas {coef[1]:.3f}, {coef[2]:.3f} (true 1.2, 0.5), R^2 {r2:.3f}")

    # 7. Weights are non-negative and sum to <= 1 for every scheme, gated too.
    px = _synthetic(1200, [0.1, 0.15, 0.25, 0.35], corr=0.4, seed=7)
    sig = pd.DataFrame({c: STRATEGIES["sma_crossover"]["func"](px[[c]].rename(
        columns={c: "Close"})) for c in px}, index=px.index)
    for wt in WEIGHTINGS:
        for s in (None, sig):
            W = _core(px, s, wt, rebalance="weekly")["weights"]
            assert W.min() >= -1e-12 and W.sum(1).max() <= 1 + 1e-9, wt
    # Risk parity really equalizes risk contributions; inverse-vol doesn't.
    win = px.pct_change().dropna().to_numpy()
    rp, cov = _target_weights(win, "risk_parity", 0.1)
    iv, _ = _target_weights(win, "inverse_vol", 0.1)
    assert np.ptp(_risk_contrib(rp, cov)) < 1e-6
    assert np.ptp(_risk_contrib(iv, cov)) > np.ptp(_risk_contrib(rp, cov))
    print("  7. weights >= 0, sum <= 1 for all schemes; risk parity contributions equal")
    print("portfolio demo: all checks passed")


if __name__ == "__main__":
    demo()
