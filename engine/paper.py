"""
paper.py: paper-trade market makers against real Kraken BTC/USD recordings.

The question: would our market maker have made money on the REAL market?
Not a simulated one, where we wrote the other traders too.

How it works. A recording (capture.py) is every change to Kraken's book plus
every trade, with timestamps. We walk through it in time order, as if live:
at each book update a strategy sees only what has happened so far and decides
where to quote. Its orders are imaginary (paper), so they can't move the real
market, and they fill ONLY when a real recorded trade reaches them:

  * Queue position. Exchanges fill orders at the same price first come, first
    served. A new order joins the BACK of the line, behind everything already
    resting at that price, so a trade first eats the orders ahead of you
    (replay.py's RestingOrder). If a trade is too small to reach you, you get
    nothing, even though "the price traded at your bid".
  * Latency. Orders and cancels take time to reach the exchange (default
    100 ms). A trade that prints before your order arrives can't fill it, and
    your old quote can still be hit until its cancel lands.
  * Fees. Kraken charges a maker fee on every fill (research.FEE_TIERS). A new
    account pays 40 bp (0.40%) per side; BTC/USD's spread is usually one tick,
    about 0.001%. So at retail fees market making loses on every fill before
    anything else happens. That's a finding, not a bug: fees decide who can be
    a market maker. The report shows each tier so you can see where it flips.

What gets measured, per strategy and recording:

  spread capture   for each fill, how far from the mid you bought (sold): the
                   reward for providing liquidity.
  inventory PnL    position x mid change while you hold inventory: the risk.
                   spread capture + inventory PnL = gross PnL, exactly.
  markouts         mid 1 s and 10 s after each fill, relative to your fill
                   price. Negative means the people who traded with you knew
                   something (adverse selection): you bought just before the
                   price fell. This is the main way market makers lose.

Three strategies, quoting over the same period of the same data:

  naive        quote at the best bid and ask, inventory limit only.
  A-S          Avellaneda-Stoikov: shade both quotes against inventory
               (backend/market_sim.py's formulas, with ticks and events).
  order-flow   shift both quotes by kappa x the order-flow forecast
               (research.py's features), fitted ONLY on the first
               train_seconds of the recording. Nobody quotes during that
               training period, so all three are scored on the same window.

    python paper.py "data/kraken_BTCUSD_*.jsonl.gz"   # analyse new recordings, print the scoreboard
    python paper.py --refresh "data/..."             # re-analyse everything (after a model change)
    python paper.py --demo                           # self-check
"""

import csv
import glob
import gzip
import json
import math
import os
import sys
from collections import deque

import numpy as np

try:
    from engine.capture import KRAKEN_PRECISION
    from engine.orderbook import BUY, SELL
    from engine.replay import RestingOrder, best_pair, ofi, queue_ahead, replay
    from engine.research import FEE_TIERS
    from engine.study import _start
except ImportError:
    from capture import KRAKEN_PRECISION
    from orderbook import BUY, SELL
    from replay import RestingOrder, best_pair, ofi, queue_ahead, replay
    from research import FEE_TIERS
    from study import _start

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.market_sim import AvellanedaStoikov  # noqa: E402  (needs the repo root on the path)

_PP = KRAKEN_PRECISION["BTC/USD"][0]
TICK = 10.0 ** -_PP   # $0.1: prices are handled as whole ticks, never floats
SIZE = 0.001          # BTC per quote, about $80: small enough to rarely move a queue
MAX_LOTS = 5          # inventory limit, in quotes: at most 0.005 BTC long or short
LATENCY = 0.100       # seconds from decision to the exchange acting on it
TRAIN_SECONDS = 900   # first 15 minutes: fit the forecast, nobody quotes
HORIZON = 10          # forecast horizon, in book events (as research.py)
WINDOW = 10           # OFI window, in book events
KAPPA = 1.0           # order-flow skew: ticks of quote shift per tick of forecast
MARKOUTS = (1, 10)    # seconds

_TIERS = {name: maker for name, (maker, _taker) in FEE_TIERS.items()}
_TIER_COL = {name: "net_" + "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()
             for name in _TIERS}
FIELDS = (["file", "bytes", "start", "strategy", "hours", "fills", "volume_btc", "max_inv_btc",
           "spread_capture", "inventory_pnl", "gross"] + list(_TIER_COL.values())
          + [f"markout_{h}s_bp" for h in MARKOUTS])


def _floor(x):
    return math.floor(x + 1e-9)      # guard against 3.9999999 from float ticks


def _ceil(x):
    return math.ceil(x - 1e-9)


# ---------------------------------------------------------------------------
# Strategies: given the market now (and the past), where to quote, in ticks.
# Each returns (bid, ask); None means "no quote on that side".
# ---------------------------------------------------------------------------
class Naive:
    name = "naive"

    def quote(self, m):
        return m["bb"], m["ba"]


class AS:
    """Avellaneda-Stoikov through market_sim's own implementation.

    That class works in ticks and steps with q in lots, so it is fed exactly
    that: mid in ticks, the recent mid history in ticks (one entry per book
    event), and position in lots (size 1).
    """
    name = "A-S"

    def __init__(self, gamma=0.1, k=1.5, horizon=5, window=50):
        self.model = AvellanedaStoikov("A-S", gamma=gamma, k=k, horizon=horizon,
                                       window=window, size=1)

    def quote(self, m):
        self.model.position = m["lots"]
        r, delta = self.model.quotes(m["mid"], m["history"])
        return _floor(r - delta / 2), _ceil(r + delta / 2)


class OrderFlow:
    """Shift both quotes toward where the order-flow forecast says the mid is going.

    Expecting a rise, the ask moves up (don't sell cheap to someone who knows)
    and the bid stays keen; expecting a fall, the reverse. With a zero
    forecast this is exactly the naive maker.
    """
    name = "order-flow"

    def __init__(self, kappa=KAPPA):
        self.kappa = kappa

    def quote(self, m):
        c = m["mid"] + self.kappa * m["forecast"]
        half = (m["ba"] - m["bb"]) / 2
        return _floor(c - half), _ceil(c + half)


STRATEGIES = (Naive, AS, OrderFlow)


# ---------------------------------------------------------------------------
# One strategy's account and orders
# ---------------------------------------------------------------------------
class Account:
    """A strategy's paper orders, position and PnL. Prices in ticks, size in BTC."""

    def __init__(self, strategy):
        self.strategy = strategy
        self.live = {BUY: None, SELL: None}      # RestingOrder on the exchange
        self.pending = {BUY: None, SELL: None}   # (price or None, time it takes effect)
        self.position = self.cash = 0.0          # BTC; cash in tick-dollars (x TICK later)
        self.spread_capture = self.inventory_pnl = 0.0
        self.volume = self.max_inv = 0.0
        self.fills = []                          # (ts, side, price_ticks, qty)

    def want(self, side, price, ts, latency):
        """Ask for `price` on `side` (None = no quote). Only a change is sent."""
        have = self.pending[side] or ((self.live[side].price, None) if self.live[side] else (None,))
        if have[0] == price:
            return
        self.pending[side] = (price, ts + latency)

    def arrive(self, ts, levels):
        """Orders and cancels whose latency has elapsed reach the exchange now."""
        for side in (BUY, SELL):
            p = self.pending[side]
            if p is None or p[1] > ts:
                continue
            self.pending[side] = None
            price = p[0]
            self.live[side] = None if price is None else RestingOrder(
                side, price, SIZE, ahead=queue_ahead(levels, side, round(price * TICK, _PP)))

    def on_trade(self, price, qty, buyer_is_maker, ts, mid):
        for side in (BUY, SELL):
            o = self.live[side]
            if o is None:
                continue
            got = o.on_trade(price, qty, buyer_is_maker)
            if got <= 0:
                continue
            sign = 1 if side == BUY else -1
            self.position += sign * got
            self.cash -= sign * got * o.price
            # Spread capture: how far on the right side of the mid we traded.
            # Kraken sends book and trade messages separately, and the book
            # update for a trade often arrives first, so `mid` can already
            # include the trade's own move. Spread capture is then net of that
            # instant move; the identity with inventory PnL is exact either way.
            self.spread_capture += sign * got * (mid - o.price)
            self.volume += got
            self.max_inv = max(self.max_inv, abs(self.position))
            self.fills.append((ts, side, o.price, got))
            if o.remaining <= 1e-12:
                self.live[side] = None           # fully filled: stop quoting until requoted


# ---------------------------------------------------------------------------
# The replay loop
# ---------------------------------------------------------------------------
def simulate(path, strategies=None, latency=LATENCY, train_seconds=TRAIN_SECONDS,
             horizon=HORIZON, window=WINDOW, peek=None):
    """Run every strategy over one recording. Returns (books, timeline).

    timeline = (ts array, mid-in-ticks array) of every book event, for markouts.
    `peek` is for the self-check only: a function (event index, mids) giving a
    forecast that is allowed to cheat, to prove the harness would notice.
    """
    books = [Account(s) for s in (strategies or [S() for S in STRATEGIES])]
    ts_list, mids, X, flows = [], [], [], deque(maxlen=window)
    t0 = start_quoting = None
    w = sd = None
    prev_pair = None
    mid = None
    for kind, msg, state in replay(path):
        ts = float(msg.get("ts", 0.0))
        if t0 is None:
            t0 = ts
            start_quoting = t0 + train_seconds
        for b in books:
            b.arrive(ts, state)

        if kind == "trade":
            if mid is not None:
                p = round(float(msg["p"]) / TICK)
                for b in books:
                    b.on_trade(p, float(msg["q"]), bool(msg["m"]), ts, mid)
            continue

        pair = best_pair(state)
        if pair is None:
            continue
        if msg.get("type") == "snapshot":
            prev_pair = None                     # a resync is a jump, not order flow
        (bp, bq), (ap, aq) = pair
        bb, ba = round(bp / TICK), round(ap / TICK)
        new_mid = (bb + ba) / 2
        if mid is not None:
            for b in books:                      # holding inventory while the mid moves
                b.inventory_pnl += b.position * (new_mid - mid)
        mid = new_mid

        # Order-flow features, identical to research.orderflow_dataset, from the past only.
        flows.append(ofi(prev_pair, pair) if prev_pair is not None else 0.0)
        prev_pair = pair
        past = mid - mids[-horizon] if len(mids) >= horizon else 0.0
        x = (sum(flows), (bq - aq) / (bq + aq), float(np.sign(past)))
        ts_list.append(ts)
        mids.append(mid)

        if ts < start_quoting:
            X.append(x)
            continue
        if w is None:
            w, sd = _fit(X, mids, horizon)
        forecast = peek(len(mids) - 1) if peek else float(np.dot(np.asarray(x) / sd, w))

        for b in books:
            m = {"bb": bb, "ba": ba, "mid": mid, "history": mids[-51:], "forecast": forecast,
                 "lots": round(b.position / SIZE)}
            bid, ask = b.strategy.quote(m)
            # Stay passive: a bid at or above the best ask would trade at once and
            # pay the taker fee instead of earning the spread.
            bid = None if bid is None else min(bid, ba - 1)
            ask = None if ask is None else max(ask, bb + 1)
            # Inventory limit: stop buying when long enough, selling when short enough.
            if b.position >= MAX_LOTS * SIZE - 1e-12:
                bid = None
            if b.position <= -MAX_LOTS * SIZE + 1e-12:
                ask = None
            b.want(BUY, bid, ts, latency)
            b.want(SELL, ask, ts, latency)

    for b in books:
        b.start, b.end = start_quoting, (ts_list[-1] if ts_list else start_quoting)
        b.final_mid = mid
    return books, (np.asarray(ts_list, float), np.asarray(mids, float))


def _fit(X, mids, horizon, ridge=1e-3):
    """Forecast of the mid's move over `horizon` events, in ticks.

    Fitted on the training window only. Rows whose label would reach past the
    window are dropped (the purge), so no quoting-period price leaks in. No
    intercept, as in research.py: an intercept learns this hour's drift,
    which is not a forecast.
    """
    n = len(X) - horizon
    if n < 50:
        return np.zeros(3), np.ones(3)
    A = np.asarray(X[:n], float)
    y = np.asarray(mids[horizon:horizon + n]) - np.asarray(mids[:n])
    sd = A.std(axis=0)
    sd[sd == 0] = 1.0
    A = A / sd
    w = np.linalg.solve(A.T @ A + ridge * n * np.eye(A.shape[1]), A.T @ y)
    return w, sd


def score(b, timeline):
    """One strategy's results as a dict of FIELDS (dollar figures in USD)."""
    gross_ticks = b.cash + b.position * (b.final_mid or 0.0)
    ts, mid = timeline
    notional = sum(p * q for _, _, p, q in b.fills) * TICK
    row = {"strategy": b.strategy.name, "hours": round(max(b.end - b.start, 0.0) / 3600, 4),
           "fills": len(b.fills), "volume_btc": round(b.volume, 8),
           "max_inv_btc": round(b.max_inv, 8),
           "spread_capture": round(b.spread_capture * TICK, 6),
           "inventory_pnl": round(b.inventory_pnl * TICK, 6), "gross": round(gross_ticks * TICK, 6)}
    for name, bp in _TIERS.items():
        row[_TIER_COL[name]] = round(gross_ticks * TICK - notional * bp / 1e4, 6)
    for h in MARKOUTS:
        outs = []
        for t, side, p, _q in b.fills:
            j = np.searchsorted(ts, t + h, side="right") - 1
            if t + h <= ts[-1]:
                sign = 1 if side == BUY else -1
                outs.append(sign * (mid[j] - p) / p * 1e4)
        row[f"markout_{h}s_bp"] = round(float(np.mean(outs)), 4) if outs else ""
    return row


def analyse(path, **kw):
    """One recording -> one row per strategy."""
    books, timeline = simulate(path, **kw)
    start = _start(path)
    base = {"file": os.path.basename(path), "bytes": os.path.getsize(path),
            "start": start.isoformat(timespec="minutes") if start else ""}
    return [dict(base, **score(b, timeline)) for b in books]


def update(paths, cache, refresh=False, quiet=False, **kw):
    """Analyse recordings not yet in `cache` (or that grew); return all rows."""
    rows = {}
    if os.path.exists(cache) and not refresh:
        with open(cache, newline="") as fh:
            for r in csv.DictReader(fh):
                rows[(r["file"], r["strategy"])] = r
    for path in sorted(paths):
        name, size = os.path.basename(path), os.path.getsize(path)
        if any(k[0] == name and int(r["bytes"]) == size for k, r in rows.items()):
            continue
        if not quiet:
            print(f"paper-trading {name} ({size / 1e6:.1f} MB)...", flush=True)
        for r in analyse(path, **kw):
            rows[(r["file"], r["strategy"])] = r
        with open(cache, "w", newline="") as fh:      # after every file, like study.py
            wr = csv.DictWriter(fh, fieldnames=FIELDS)
            wr.writeheader()
            wr.writerows(sorted(rows.values(), key=lambda r: (r["start"], r["file"], r["strategy"])))
    return sorted(rows.values(), key=lambda r: (r["start"], r["file"], r["strategy"]))


def _ci(per_hour, n_boot=10_000, seed=0):
    """Mean and 95% bootstrap interval over recordings."""
    x = np.asarray(per_hour, float)
    if len(x) < 3:
        return float(x.mean()), None
    rng = np.random.default_rng(seed)
    boots = rng.choice(x, (n_boot, len(x))).mean(axis=1)
    return float(x.mean()), (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def report(rows):
    if not rows:
        print("no recordings yet: run capture.py --live kraken (or the scheduled workflow) first")
        return
    best, worst = list(_TIER_COL.items())[-1], list(_TIER_COL.items())[0]
    print(f"\n{'start':17}{'strategy':>11}{'fills':>7}{'spread $':>10}{'inventory $':>12}"
          f"{'gross $':>10}{'mark 1s bp':>11}{'mark 10s bp':>12}{'net ' + worst[0][:11] + ' $':>18}")
    fmt = lambda v: f"{float(v):+.4f}" if v != "" else "n/a"
    for r in rows:
        print(f"{r['start']:17}{r['strategy']:>11}{int(r['fills']):7}{float(r['spread_capture']):+10.3f}"
              f"{float(r['inventory_pnl']):+12.3f}{float(r['gross']):+10.3f}"
              f"{fmt(r['markout_1s_bp']):>11}{fmt(r['markout_10s_bp']):>12}{float(r[worst[1]]):+18.3f}")

    print(f"\ntotals across recordings, USD per hour of quoting (mean, 95% bootstrap interval over recordings):")
    for strat in sorted({r["strategy"] for r in rows}):
        grp = [r for r in rows if r["strategy"] == strat and float(r["hours"]) > 0]
        if not grp:
            continue
        line = f"  {strat:11} {len(grp)} recordings, {sum(int(r['fills']) for r in grp)} fills"
        for name, col in [("gross", "gross")] + list(((n, c) for n, c in _TIER_COL.items())):
            mean, ci = _ci([float(r[col]) / float(r["hours"]) for r in grp])
            line += f"\n      {name:16}{mean:+9.3f}" + (f"  [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else
                                                       "  (interval needs 3+ recordings)")
        print(line)
    print(f"\nSize {SIZE} BTC per quote, inventory limit {MAX_LOTS} quotes, {LATENCY * 1000:.0f} ms "
          f"latency, first {TRAIN_SECONDS // 60} min of each recording used to fit the forecast.")
    print("Fills come only from real recorded trades that reach us through the queue. "
          f"At the '{worst[0]}' fee ({_TIERS[worst[0]]} bp) market making loses on every fill: "
          "the spread on BTC/USD is about 0.001%, the fee 0.40%. That is the finding, not a bug; "
          f"see the '{best[0]}' row for what the strategy itself is worth. Negative markouts are "
          "adverse selection: the traders who hit our quotes knew where the price was going.")


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
def _write(path, records):
    """A tiny hand-made recording in capture.py's format."""
    with gzip.open(path, "wt") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _snapshot(ts, bids, asks):
    return {"type": "snapshot", "ts": ts, "lastUpdateId": 0, "bids": bids, "asks": asks}


def _trade(ts, price, qty, seller_aggressor):
    # m = buyer_is_maker: True when a SELLER crossed the spread and hit the bids
    return {"type": "trade", "ts": ts, "p": price, "q": qty, "m": seller_aggressor}


def demo():
    import tempfile
    try:
        from engine.research import synthetic_kraken_capture
    except ImportError:
        from research import synthetic_kraken_capture

    with tempfile.TemporaryDirectory() as d:
        # 1. Queue position: 0.005 BTC is already bid at 50000.0 when we join.
        #    A 0.003 sell doesn't reach us; the next 0.003 eats the last 0.002
        #    ahead of us and fills 0.001, our whole order.
        q = os.path.join(d, "queue.jsonl.gz")
        _write(q, [_snapshot(1.0, [["50000.0", "0.005"]], [["50000.1", "0.005"]]),
                   _trade(2.0, "50000.0", "0.003", True),
                   _trade(3.0, "50000.0", "0.003", True),
                   _trade(4.0, "50000.1", "0.010", True)])   # seller at the ask: not our bid's business
        books, _ = simulate(q, [Naive()], latency=0.0, train_seconds=0)
        assert [(t, s) for t, s, _, _ in books[0].fills] == [(3.0, BUY)], books[0].fills
        assert abs(books[0].position - SIZE) < 1e-12

        # 2. Latency: a trade that prints before our order arrives can't fill it.
        lat = os.path.join(d, "latency.jsonl.gz")
        _write(lat, [_snapshot(1.0, [["50000.0", "0.005"]], [["50000.1", "0.005"]]),
                     _trade(1.05, "50000.0", "0.010", True)])
        assert len(simulate(lat, [Naive()], latency=0.0, train_seconds=0)[0][0].fills) == 1
        assert len(simulate(lat, [Naive()], latency=0.1, train_seconds=0)[0][0].fills) == 0

        # 3. On a synthetic Kraken recording: exact PnL identity and fee arithmetic.
        syn = os.path.join(d, "kraken_BTCUSD_20261101_030000.jsonl.gz")
        synthetic_kraken_capture(syn, n=6000, signal=True, seed=4)
        books, tl = simulate(syn, train_seconds=60)
        for b in books:
            r = score(b, tl)
            assert r["fills"] > 0, f"{r['strategy']} never filled"
            assert abs(r["spread_capture"] + r["inventory_pnl"] - r["gross"]) < 1e-6, r
            notional = sum(p * qty for _, _, p, qty in b.fills) * TICK
            for name, bp in _TIERS.items():
                assert abs(r[_TIER_COL[name]] - (r["gross"] - notional * bp / 1e4)) < 1e-6
            assert all(t >= b.start for t, *_ in b.fills), "filled during the training window"
            assert b.max_inv <= MAX_LOTS * SIZE + 1e-12, "inventory limit broken"

        # 4. No lookahead: cutting the recording off at time T changes nothing
        #    that happened before T (minus latency). If a strategy, or the
        #    forecast's training, saw the future, the two runs would differ.
        recs = []
        with gzip.open(syn, "rt") as fh:
            recs = [json.loads(line) for line in fh]
        cut_ts = recs[len(recs) * 3 // 4]["ts"]
        cut = os.path.join(d, "cut.jsonl.gz")
        _write(cut, [r for r in recs if r["ts"] <= cut_ts])
        short, _ = simulate(cut, train_seconds=60)
        for full_b, cut_b in zip(books, short):
            early = [f for f in full_b.fills if f[0] <= cut_ts]
            assert early == cut_b.fills, f"{full_b.strategy.name} changed its past when the future was removed"

        # ...and the harness does reward knowing the future, so that check has
        # teeth: a forecast that peeks at the real next move avoids being
        # picked off, and its 1 s markouts beat the honest forecast's.
        _, mids_all = tl
        cheat = lambda i: float(mids_all[min(i + HORIZON, len(mids_all) - 1)] - mids_all[i]) * 2
        cb, ctl = simulate(syn, [OrderFlow()], train_seconds=60, peek=cheat)
        honest = score(books[2], tl)["markout_1s_bp"]
        assert score(cb[0], ctl)["markout_1s_bp"] > honest, "peeking didn't help: harness can't see information"

        # 5. Caching: a second update() reads the CSV instead of re-running.
        cache = os.path.join(d, "paper.csv")
        rows = update([syn], cache, quiet=True, train_seconds=60)
        assert {r["strategy"] for r in rows} == {"naive", "A-S", "order-flow"}
        global analyse
        real, calls = analyse, []
        analyse = lambda p, **kw: calls.append(p) or real(p, **kw)
        try:
            again = update([syn], cache, quiet=True, train_seconds=60)
        finally:
            analyse = real
        assert not calls and len(again) == 3 and again[0]["gross"] == str(rows[0]["gross"])

    print("ok — paper: queue position, latency, PnL identity, fees, inventory limit, "
          "no lookahead (and peeking would be noticed), cache")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        args = [a for a in sys.argv[1:] if not a.startswith("--")]
        paths = [p for a in (args or ["data/kraken_BTCUSD_*.jsonl.gz"]) for p in glob.glob(a)]
        cache = os.path.join(os.path.dirname(paths[0]) if paths else "data", "paper.csv")
        report(update(paths, cache, refresh="--refresh" in sys.argv))
