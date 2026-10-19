"""
forecast.py — a short-horizon order-flow forecast, and a market maker that uses it.

The question a market maker lives or dies by is not "what is the price?" but
"where is the price going in the next few seconds?". If buyers are queueing
up, the mid is about to tick up, and a maker still selling at the old price is
handing money to whoever is better informed. That loss is ADVERSE SELECTION,
and it is the main cost of making markets.

This module does three things on the simulated market in market_sim.py:

  1. OrderFlowForecaster  predicts the mid change over the next h steps from
     four standard order-flow features, using only information available at
     step t, refitted in a walk-forward loop that never peeks at the future.
  2. ForecastMaker        an Avellaneda-Stoikov maker whose reservation price
     leans toward the forecast:  r = s + kappa*E[ds] - q*gamma*sigma^2*(T-t).
  3. run()                the same market (same seed, same background agents)
     three times, with the maker slot naive / A-S / ForecastMaker, scored by
     PnL, its exact spread-vs-inventory decomposition, and MARKOUTS.

Features (all at step t, from the book AFTER step t's matching, and past trades):
  ofi             Order Flow Imbalance summed over the last `window` steps
                  (Cont, Kukanov & Stoikov 2014), using engine/replay.ofi, the
                  same definition the real-data study in engine/research.py uses.
  imbalance       top-of-book size imbalance (bid_qty - ask_qty)/(bid_qty + ask_qty).
  micro_minus_mid microprice minus mid, in ticks (Stoikov 2018).
  trade_flow      signed aggressor volume over the last `window` steps
                  (+ = buyers lifting offers).

numpy only. Run `python -m backend.forecast` for the self-check.
"""

import json
import math
import time
from collections import deque

import numpy as np

try:
    from backend.market_sim import (TICK, AvellanedaStoikov, MarketMaker, Simulation,
                                    _maker_summary, default_agents)
    from engine.orderbook import BUY, SELL
    from engine.replay import ofi
    from engine.research import information_coefficient
except ImportError:                        # running this file directly
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend.market_sim import (TICK, AvellanedaStoikov, MarketMaker, Simulation,
                                    _maker_summary, default_agents)
    from engine.orderbook import BUY, SELL
    from engine.replay import ofi
    from engine.research import information_coefficient

FEATURES = ("ofi", "imbalance", "micro_minus_mid", "trade_flow")
HORIZONS = (1, 5, 10)
MARKOUT_LAGS = (1, 5, 20)
MAX_STEPS = 5_000
CONTROL_SEED_OFFSET = 1_000      # negative control: features from seed + this


def top_of_book(book):
    """((bid_px, bid_qty), (ask_px, ask_qty)) in ticks/shares, or None if one-sided."""
    b, a = book.depth(BUY, 1), book.depth(SELL, 1)
    if not b or not a:
        return None
    return (b[0][0], b[0][1]), (a[0][0], a[0][1])


def mid_of(top, fallback):
    """Exact (float) mid in ticks. The simulator's own `mid()` floors to a
    whole tick; for a forecast target that rounding is just noise."""
    if top is None:
        return float(fallback)
    return (top[0][0] + top[1][0]) / 2


# ---------------------------------------------------------------------------
# The forecaster
# ---------------------------------------------------------------------------
class OrderFlowForecaster:
    """Online ridge regression of the h-step mid change on order-flow features.

    Feed it one row per step with observe(top, flow, mid); it returns the
    forecast for mid[t+h] - mid[t] in ticks, using only rows 0..t.

    Walk-forward fitting. Every `retrain_every` steps the model is refitted on
    past rows (expanding window, or the last `max_train` rows if set). Row i's
    target is mid[i+h] - mid[i], so at refit time T it is only KNOWN for
    i <= T - h. The last h rows are therefore left out of the fit: that is the
    PURGE GAP. Including them would mean training on mid prices from after T,
    i.e. on the future, and the backtest would show skill a live trader could
    never have. It also stops overlapping targets from leaking: rows T-h+1..T
    share future price moves with the rows we are about to forecast.

    Features are standardized with the TRAINING rows' mean and std (never the
    full sample), and there is no intercept: at average order flow the model
    forecasts no move, rather than extrapolating whatever drift the training
    window happened to have.
    """

    def __init__(self, horizon=5, window=5, retrain_every=100, min_train=200,
                 ridge=10.0, max_train=None):
        if horizon < 1 or window < 1 or retrain_every < 1:
            raise ValueError("horizon, window and retrain_every must be >= 1")
        self.h, self.window = int(horizon), int(window)
        self.retrain_every, self.min_train = int(retrain_every), int(min_train)
        self.ridge, self.max_train = float(ridge), max_train
        self.X, self.mid = [], []
        self.coef = self.mu = self.sd = None
        self.fits = []                      # (T, first_row, last_row) per refit
        self._prev = None
        self._ofi, self._flow = deque(), deque()
        self._ofi_sum = self._flow_sum = 0.0

    def _features(self, top, flow):
        e = ofi(self._prev, top) if (self._prev is not None and top is not None) else 0.0
        self._prev = top
        for q, v, attr in ((self._ofi, e, "_ofi_sum"), (self._flow, float(flow), "_flow_sum")):
            q.append(v)
            setattr(self, attr, getattr(self, attr) + v)
            if len(q) > self.window:
                setattr(self, attr, getattr(self, attr) - q.popleft())
        if top is None:
            return [self._ofi_sum, 0.0, 0.0, self._flow_sum]
        (bp, bq), (ap, aq) = top
        micro = (bq * ap + aq * bp) / (bq + aq)
        return [self._ofi_sum, (bq - aq) / (bq + aq), micro - (bp + ap) / 2, self._flow_sum]

    def observe(self, top, flow, mid):
        """Add step t's information; return the forecast made AT t."""
        x = self._features(top, flow)
        self.X.append(x)
        self.mid.append(float(mid))
        T = len(self.X) - 1
        if (T + 1) % self.retrain_every == 0:
            self._fit(T)
        return self.predict(x)

    def _fit(self, T):
        last = T - self.h                   # newest row whose target is known at T
        first = 0 if self.max_train is None else max(0, last - int(self.max_train) + 1)
        if last - first + 1 < self.min_train:
            return
        X = np.asarray(self.X[first:last + 1], float)
        m = np.asarray(self.mid, float)
        y = m[first + self.h:last + self.h + 1] - m[first:last + 1]   # uses mid[<= T] only
        mu, sd = X.mean(0), X.std(0)
        sd[sd == 0] = 1.0
        Z = (X - mu) / sd
        self.coef = np.linalg.solve(Z.T @ Z + self.ridge * np.eye(Z.shape[1]), Z.T @ y)
        self.mu, self.sd = mu, sd
        self.fits.append((T, first, last))

    def predict(self, x):
        if self.coef is None:
            return 0.0
        return float(((np.asarray(x, float) - self.mu) / self.sd) @ self.coef)

    @property
    def ready(self):
        return self.coef is not None


def walk_forward(rows, horizon, mids=None, **kw):
    """Run the forecaster over recorded rows (top, flow, mid).

    `mids` overrides the target series (the negative control pairs one
    market's features with another market's prices). Returns (forecasts,
    target mids, ready mask, forecaster)."""
    fc = OrderFlowForecaster(horizon=horizon, **kw)
    mids = [r[2] for r in rows] if mids is None else mids
    f, ready = np.zeros(len(rows)), np.zeros(len(rows), bool)
    for t, (top, flow, _m) in enumerate(rows):
        f[t] = fc.observe(top, flow, mids[t])
        ready[t] = fc.ready
    return f, np.asarray(mids, float), ready, fc


def score(f, mids, ready, horizon):
    """Out-of-sample forecast quality: every forecast made by a fitted model
    whose h-step outcome is in the sample.

      r2        1 - SSE(model)/SSE(zero forecast). The zero forecast ("the
                price won't move") is the baseline to beat; R^2 < 0 means
                worse than saying nothing.
      ic        Spearman rank correlation of forecast vs realised change
                (engine/research.information_coefficient).
      ic_t      ic * sqrt(n / h). Targets h steps long overlap, so there are
                only about n/h independent observations; using n would make
                noise look significant. Same convention as research.py.
      hit_rate  sign agreement on steps where the mid actually moved.
    """
    h = int(horizon)
    n_all = len(mids)
    idx = np.flatnonzero(ready[: n_all - h])
    if len(idx) < 20:
        return {"n": int(len(idx)), "r2": None, "ic": None, "ic_t": None, "hit_rate": None}
    y = mids[idx + h] - mids[idx]
    p = f[idx]
    sse0 = float((y ** 2).sum())
    r2 = 1 - float(((y - p) ** 2).sum()) / sse0 if sse0 > 0 else 0.0
    ic = information_coefficient(p, y)
    called = (y != 0) & (p != 0)
    return {
        "n": int(len(idx)),
        "r2": round(r2, 4),
        "ic": round(ic, 4),
        "ic_t": round(ic * math.sqrt(max(len(idx) / h, 1.0)), 2),
        "hit_rate": round(float((np.sign(p[called]) == np.sign(y[called])).mean()), 4)
        if called.any() else None,
        "moved_frac": round(float((y != 0).mean()), 4),
    }


# ---------------------------------------------------------------------------
# Makers and markouts
# ---------------------------------------------------------------------------
class _FillLog:
    """Mixin: remember every fill with the step it happened in, for markouts.
    The driver sets `clock` to the step number before each step."""
    clock = 0

    def on_fill(self, side, price_ticks, qty):
        if not hasattr(self, "fill_log"):
            self.fill_log = []
        self.fill_log.append((self.clock, side, price_ticks, qty))
        super().on_fill(side, price_ticks, qty)


class LoggedMaker(_FillLog, MarketMaker):
    pass


class LoggedAS(_FillLog, AvellanedaStoikov):
    pass


class ForecastMaker(_FillLog, AvellanedaStoikov):
    """Avellaneda-Stoikov with a view: the reservation price leans toward the
    order-flow forecast,

        r = s + kappa * E[s_{t+h} - s_t] - q * gamma * sigma^2 * (T - t)

    with the forecast in ticks. If buyers are about to push the mid up, the
    maker raises both quotes: it bids more eagerly (wants to own the stock
    before it rises) and offers higher (stops selling cheap to informed
    buyers). kappa = 0 is plain A-S. Everything else is inherited unchanged:
    spread formula, inventory limit, max_offset cap, passivity clamps, and the
    exact spread-capture + inventory PnL decomposition.

    `forecast` is set by the driver from the forecaster's output at the END of
    the previous step, so the maker only ever acts on information it had.
    """

    def __init__(self, name, kappa=1.0, **kw):
        super().__init__(name, **kw)
        self.kappa = float(kappa)
        self.forecast = 0.0

    def quotes(self, mid, history):
        r, delta = super().quotes(mid, history)
        return r + self.kappa * self.forecast, delta


def markouts(fills, mids, lags=MARKOUT_LAGS):
    """Markouts: how the market moved after each of our fills.

    This is the standard adverse-selection measure. A fill at step j happens
    against the mid mids[j-1] (the start of that step); +k is mids[j-1+k].
      markout   side * (mid[+k] - fill price) per share: the PnL of the fill
                marked k steps later. It starts at the half-spread we earned
                and decays if the price moves against us.
      mid_move  side * (mid[+k] - mid at the fill): the pure price move. A
                negative number means we tended to buy just before falls and
                sell just before rises, i.e. we traded with better-informed
                flow. A maker with a good short-horizon forecast should push
                this toward zero or above.
    side is +1 for our buys, -1 for our sells, so positive = good for the
    maker. Volume-weighted, in ticks per share. Fills too close to the end of
    the run to have a +k mid are left out of that lag.
    """
    out_m, out_d = {}, {}
    for k in lags:
        num_m = num_d = qty = 0.0
        for j, side, price, q in fills:
            if j - 1 + k >= len(mids) or j < 1:
                continue
            sgn = 1 if side == BUY else -1
            num_m += sgn * (mids[j - 1 + k] - price) * q
            num_d += sgn * (mids[j - 1 + k] - mids[j - 1]) * q
            qty += q
        out_m[str(k)] = round(num_m / qty, 4) if qty else None
        out_d[str(k)] = round(num_d / qty, 4) if qty else None
    return {"markout_ticks": out_m, "mid_move_ticks": out_d}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def simulate(maker, seed, steps, forecaster=None):
    """One market: default_agents() + `maker`. Records the order-flow rows
    (top of book, signed trade flow, mid) after every step. If `forecaster`
    is given it observes each step and feeds `maker.forecast` for the next."""
    sim = Simulation(agents=default_agents() + [maker], seed=seed)
    start_top = top_of_book(sim.book)
    mids = [mid_of(start_top, sim.mid())]
    rows, inv, pnl, fc_path = [], [], [], []
    for i in range(steps):
        maker.clock = i + 1
        sim.step()
        top = top_of_book(sim.book)
        m = mid_of(top, sim.history[-1])
        mids.append(m)
        rows.append((top, sim.last_flow, m))
        if forecaster is not None:
            maker.forecast = forecaster.observe(top, sim.last_flow, m)
            fc_path.append(maker.forecast)
        inv.append(maker.position)
        pnl.append(maker.equity(sim.history[-1]) - maker.start_cash)
    return {"sim": sim, "rows": rows, "mids": mids, "inv": inv, "pnl": pnl,
            "forecast": fc_path}


def _summary(maker, rec):
    s = _maker_summary(maker, rec["inv"], rec["sim"].history[-1])
    s.update(markouts(getattr(maker, "fill_log", []), rec["mids"]))
    return s


def clamp_steps(steps):
    """API guard: 500 <= steps <= MAX_STEPS, so a request can't run for minutes."""
    return max(500, min(int(steps), MAX_STEPS))


def run(steps=2000, seed=7, horizon=5, kappa=1.0, gamma=0.1, k=1.5, window=5,
        retrain_every=100, inventory_limit=200, size=10, naive_spread=4, max_points=300):
    """Order-flow forecast quality, and naive vs A-S vs ForecastMaker on one market.

    Inputs are bounded so an API call stays fast (steps <= 5000, about 4 s).
    The forecast is evaluated out of sample on the A-S run (where nobody uses
    it, so it is a pure observer), at horizons 1, 5, 10, next to a NEGATIVE
    CONTROL: features from a different seed's market paired with this
    market's prices. The control has the same feature statistics and the same
    target statistics but no causal link, so it must show no skill; if it
    did, the evaluation itself would be leaking.
    """
    if gamma <= 0 or k <= 0:
        raise ValueError("gamma and k must be positive")
    steps = clamp_steps(steps)
    horizon = max(1, min(int(horizon), 50))
    window = max(1, min(int(window), 100))
    retrain_every = max(20, min(int(retrain_every), 1_000))
    kappa = max(-10.0, min(float(kappa), 10.0))
    max_points = max(10, min(int(max_points), 1_000))
    fkw = {"window": window, "retrain_every": retrain_every}
    mk = {"size": size, "max_inv": inventory_limit}
    ask = {"gamma": gamma, "k": k, "horizon": horizon, **mk}

    naive = LoggedMaker("Naive", spread=naive_spread, **mk)
    a_s = LoggedAS("A-S", **ask)
    fm = ForecastMaker("Forecast", kappa=kappa, **ask)
    rec_n = simulate(naive, seed, steps)
    rec_a = simulate(a_s, seed, steps)
    live_fc = OrderFlowForecaster(horizon=horizon, **fkw)
    rec_f = simulate(fm, seed, steps, forecaster=live_fc)
    ctrl = simulate(LoggedAS("A-S", **ask), seed + CONTROL_SEED_OFFSET, steps)

    by_h = {}
    for h in sorted(set(HORIZONS) | {horizon}):
        f, mids, ready, fc = walk_forward(rec_a["rows"], h, **fkw)
        if h == horizon:
            path_fc = (f, mids, ready)
        fn, _, ready_n, _ = walk_forward(ctrl["rows"], h, mids=mids, **fkw)
        by_h[str(h)] = {
            "model": score(f, mids, ready, h),
            "negative_control": score(fn, mids, ready_n, h),
            "coef_standardized": {n: round(float(c), 4) for n, c in zip(FEATURES, fc.coef)}
            if fc.ready else None,
        }
    # The forecast as the ForecastMaker actually used it, live, in its own market.
    live_mids = np.asarray([r[2] for r in rec_f["rows"]], float)
    first_fit = live_fc.fits[0][0] if live_fc.fits else steps
    live = score(np.asarray(rec_f["forecast"]), live_mids, np.arange(steps) >= first_fit, horizon)

    # Downsampled paths. The forecast-vs-realised pair is from the A-S run at
    # the maker's horizon: forecast made at t, and what then happened by t+h.
    f, mids, ready = path_fc
    stride = max(1, -(-steps // max_points))
    pick = list(range(0, steps, stride))
    realized = [round(mids[t + horizon] - mids[t], 2) if t + horizon < steps else None
                for t in pick]
    px = lambda v: round(v * TICK, 4)
    return {
        "params": {"steps": steps, "seed": seed, "horizon": horizon, "kappa": kappa,
                   "gamma": gamma, "k": k, "window": window, "retrain_every": retrain_every,
                   "inventory_limit": inventory_limit, "size": size,
                   "naive_spread": naive_spread, "features": list(FEATURES),
                   "control_seed": seed + CONTROL_SEED_OFFSET},
        "forecast": {"by_horizon": by_h, "live_in_forecast_maker_run": live},
        "makers": {"naive": _summary(naive, rec_n),
                   "avellaneda_stoikov": _summary(a_s, rec_a),
                   "forecast_maker": _summary(fm, rec_f)},
        "paths": {
            "step": [t + 1 for t in pick],
            "mid": [px(rec_a["mids"][t + 1]) for t in pick],
            "forecast_ticks": [round(float(f[t]), 3) if ready[t] else None for t in pick],
            "realized_ticks": realized,
            "pnl": {name: [round(r["pnl"][t], 2) for t in pick]
                    for name, r in (("naive", rec_n), ("avellaneda_stoikov", rec_a),
                                    ("forecast_maker", rec_f))},
            "inventory": {name: [r["inv"][t] for t in pick]
                          for name, r in (("naive", rec_n), ("avellaneda_stoikov", rec_a),
                                          ("forecast_maker", rec_f))},
        },
    }


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
def demo():
    t_start = time.perf_counter()

    # Markout signs on a hand-built example. Mids 100,101,102,...: rising.
    mids = [100.0 + i for i in range(30)]
    good = markouts([(1, BUY, 99, 10)], mids, lags=(1, 5))     # bought below, price rose
    bad = markouts([(1, SELL, 100, 10)], mids, lags=(1, 5))    # sold at mid, price rose
    assert good["markout_ticks"] == {"1": 2.0, "5": 6.0}, good
    assert good["mid_move_ticks"] == {"1": 1.0, "5": 5.0}, good
    assert bad["markout_ticks"] == {"1": -1.0, "5": -5.0}, bad
    assert bad["mid_move_ticks"] == {"1": -1.0, "5": -5.0}, bad
    edge = markouts([(29, BUY, 100, 1)], mids, lags=(1, 5))    # no +5 mid yet
    assert edge["markout_ticks"]["5"] is None and edge["markout_ticks"]["1"] == 29.0

    # Recorded market for the forecaster checks.
    rec = simulate(LoggedAS("A-S"), seed=7, steps=1500)
    rows = rec["rows"]

    # No lookahead: scramble everything after t0; forecasts up to t0 must not move.
    h, t0 = 5, 900
    f1, _, _, fc1 = walk_forward(rows, h)
    rng = np.random.default_rng(0)
    scrambled = rows[:t0 + 1] + [(((9_000, 7), (9_001, 3)), int(rng.integers(-50, 50)),
                                   float(rng.integers(5_000, 15_000))) for _ in rows[t0 + 1:]]
    f2, _, _, _ = walk_forward(scrambled, h)
    assert np.array_equal(f1[:t0 + 1], f2[:t0 + 1]), "forecast at t used data after t"
    assert not np.array_equal(f1[t0 + 1:], f2[t0 + 1:]), "scramble had no effect (test is blind)"

    # Purge gap: every fit at time T used only rows whose h-step target ended by T.
    assert fc1.fits, "never fitted"
    for T, first, last in fc1.fits:
        assert last + h <= T, (T, last)
    # and the gap is exactly h, not more (no data thrown away for nothing)
    assert all(T - last == h for T, _, last in fc1.fits)

    # Positive control: a market where trade flow really does move the next
    # mid must be detected, or a "no skill" result would mean nothing.
    rng = np.random.default_rng(1)
    flow = rng.integers(-20, 21, 2000)
    m = np.cumsum(np.r_[0, 0.3 * flow[:-1] + rng.normal(0, 2, 1999)]) + 10_000
    syn = [(None, int(flow[t]), float(m[t])) for t in range(2000)]
    fp, mp, rp, _ = walk_forward(syn, 1, window=1)
    pos = score(fp, mp, rp, 1)
    assert pos["ic_t"] > 10, f"positive control not detected: {pos}"

    # Negative control: another seed's features vs this seed's prices: no skill.
    worst = 0.0
    for sd in (7, 8, 9):
        r_a = simulate(LoggedAS("A-S"), seed=sd, steps=1500)["rows"]
        r_b = simulate(LoggedAS("A-S"), seed=sd + CONTROL_SEED_OFFSET, steps=1500)["rows"]
        for hh in HORIZONS:
            mids_a = [r[2] for r in r_a]
            fn, mn, rn, _ = walk_forward(r_b, hh, mids=mids_a)
            s = score(fn, mn, rn, hh)
            worst = max(worst, abs(s["ic_t"]))
    assert worst < 3.0, f"negative control shows 'skill' (|IC t| = {worst}): evaluation leaks"

    # Full run: identity for all three makers, reproducible, bounded, JSON-ready.
    t0_run = time.perf_counter()
    out = run()
    t_run = time.perf_counter() - t0_run
    for name, sm in out["makers"].items():
        assert sm["identity_error"] < 1e-6, f"{name}: total != spread + inventory"
        assert sm["fills"] > 50, f"{name} barely traded"
        assert set(sm["markout_ticks"]) == {str(k) for k in MARKOUT_LAGS}
    assert out["makers"]["avellaneda_stoikov"]["inventory_std"] < \
        out["makers"]["naive"]["inventory_std"], "A-S should hold inventory tighter"
    size = len(json.dumps(out))
    assert size < 100_000, f"payload too big: {size} bytes"
    assert run(steps=600, seed=3) == run(steps=600, seed=3), "run must be reproducible"
    assert run(steps=600, seed=3) != run(steps=600, seed=4), "seeds should differ"
    assert clamp_steps(10**9) == MAX_STEPS and clamp_steps(10) == 500
    # kappa = 0 must be exactly A-S: same quotes, same fills, same market.
    z = run(steps=600, seed=5, kappa=0.0)["makers"]
    assert z["forecast_maker"] == z["avellaneda_stoikov"], "kappa=0 should reduce to A-S"
    for bad_kw in ({"gamma": 0}, {"k": -1}):
        try:
            run(**bad_kw)
            raise AssertionError(f"accepted {bad_kw}")
        except ValueError:
            pass
    elapsed = time.perf_counter() - t_start

    fq = out["forecast"]["by_horizon"]
    print(f"ok — markout signs, no lookahead, purge gap = h, positive control "
          f"IC t {pos['ic_t']}, negative control max |IC t| {worst:.2f} (3 seeds x 3 horizons)")
    print(f"ok — run() {t_run:.2f}s, payload {size / 1024:.1f} KB, reproducible; "
          f"demo total {elapsed:.1f}s")
    print("   horizon  model: R2   IC     IC_t   hit  | control: R2   IC     IC_t")
    for hh, v in fq.items():
        mo, nc = v["model"], v["negative_control"]
        print(f"   h={hh:<5s}        {mo['r2']:+.3f} {mo['ic']:+.3f} {mo['ic_t']:+7.2f} "
              f"{mo['hit_rate']:.3f} |          {nc['r2']:+.3f} {nc['ic']:+.3f} {nc['ic_t']:+6.2f}"
              f"   coef {v['coef_standardized']}")
    print(f"   live (ForecastMaker run, h={out['params']['horizon']}): "
          f"{out['forecast']['live_in_forecast_maker_run']}")
    for name, sm in out["makers"].items():
        print(f"   {name:19s} PnL {sm['total_pnl']:9.2f} = spread {sm['spread_capture']:9.2f}"
              f" + inv {sm['inventory_pnl']:9.2f}; inv std {sm['inventory_std']:6.2f}, "
              f"fills {sm['fills']}, /fill {sm['pnl_per_fill']:+.4f}, "
              f"markout {sm['markout_ticks']}, mid move {sm['mid_move_ticks']}")


if __name__ == "__main__":
    demo()
