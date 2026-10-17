"""
market_sim.py — a market that produces its own prices.

The Arena replays recorded history: the fighters watch a price series that was
fixed before they existed, so five bots all screaming "buy" move nothing. This
module is the opposite. Agents post real orders into the matching engine in
engine/orderbook.py, those orders match against each other, and the price is
whatever they trade at. Nobody supplies it.

That makes it a closed economy, which gives two invariants worth more than any
amount of plausible-looking output:

    sum of all positions == 0      (every buy is somebody's sell)
    sum of all cash     == constant (money moves between agents, never appears)

If either drifts, the engine is inventing shares or money and every result
downstream is fiction. Both are asserted after every single step.

Four agent species, because a market of one strategy is not a market:

  MarketMaker   quotes both sides, earns the spread, and skews its quotes to
                shed inventory. This is the agent that supplies liquidity, so
                without it the book is empty and nothing trades.
  Momentum      buys strength, sells weakness. Destabilizing: it amplifies moves.
  MeanReverter  fades moves back toward a slow average. Stabilizing.
  Noise         random orders. Not filler: real markets have uninformed flow,
                and without it the informed agents have nobody to trade against.

Prices are INTEGER TICKS throughout (1 tick = 1 cent), converted to dollars
only at the boundary.

Run `python backend/market_sim.py` for the self-check.
"""

import math
import random
import time

# Book comes from fastbook: the Rust engine when its Python extension is built,
# otherwise the Python one. Both implement the same portable interface (SPEC
# 7.1) and produce identical events, which engine/test_pyo3.py checks.
try:
    from engine.fastbook import Book
    from engine.orderbook import BUY, SELL
except ImportError:                        # running this file directly
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from engine.fastbook import Book
    from engine.orderbook import BUY, SELL

TICK = 0.01          # dollars per tick
START_PRICE = 10_000  # $100.00 in ticks
START_CASH = 100_000  # dollars


class Agent:
    """Base: tracks cash and inventory, and is filled by the exchange."""

    def __init__(self, name, cash=START_CASH):
        self.name = name
        self.cash = float(cash)
        self.start_cash = float(cash)  # PnL is measured against THIS agent's
        self.position = 0              # own starting cash, not a global constant
        self.live: list[int] = []      # resting order ids

    def on_fill(self, side, price_ticks, qty):
        value = price_ticks * TICK * qty
        if side == BUY:
            self.cash -= value
            self.position += qty
        else:
            self.cash += value
            self.position -= qty

    def equity(self, mark_ticks):
        return self.cash + self.position * mark_ticks * TICK

    def act(self, book, mid, history, rnd):
        """Return a list of (kind, side, price_ticks, qty). kind: limit|market."""
        return []


class MarketMaker(Agent):
    """Quotes both sides around its fair-value estimate.

    The inventory skew is the whole job. A maker that quotes symmetrically
    accumulates whatever the market is dumping and eventually blows up, so as
    inventory grows long it shades both quotes down to encourage selling.
    """

    def __init__(self, name, spread=4, size=10, max_inv=200, **kw):
        super().__init__(name, **kw)
        self.spread, self.size, self.max_inv = spread, size, max_inv
        # PnL decomposition, kept in integer ticks*shares so it is exact.
        # A maker's profit has exactly two sources:
        #   spread capture: buying below (selling above) the fair mid at the
        #                   moment of the fill. This is the reward for
        #                   providing liquidity.
        #   inventory PnL:  the mid moving while you hold a position. This is
        #                   the risk you carry as a side effect, and it can
        #                   easily swamp the spread you earned.
        # `mark` is the mid observed at the start of the current step, before
        # anyone submitted orders (Simulation.step sets it).
        self.mark = None
        self.spread_ticks = 0
        self.inv_ticks = 0
        self.fills = 0

    def on_fill(self, side, price_ticks, qty):
        super().on_fill(side, price_ticks, qty)
        if self.mark is not None:
            edge = (self.mark - price_ticks) if side == BUY else (price_ticks - self.mark)
            self.spread_ticks += edge * qty
        self.fills += 1

    def pnl_parts(self):
        """Dollar decomposition. total == spread + inventory exactly (see step)."""
        return {"spread": self.spread_ticks * TICK, "inventory": self.inv_ticks * TICK}

    def act(self, book, mid, history, rnd):
        skew = int(self.spread * (self.position / self.max_inv)) if self.max_inv else 0
        bid = mid - self.spread - skew
        ask = mid + self.spread - skew
        orders = []
        if self.position < self.max_inv:
            orders.append(("limit", BUY, bid, self.size))
        if self.position > -self.max_inv:
            orders.append(("limit", SELL, ask, self.size))
        return orders


class AvellanedaStoikov(MarketMaker):
    """Avellaneda & Stoikov (2008), "High-frequency trading in a limit order book".

    The problem a market maker faces: quote tight and you get filled a lot
    (more spread captured), but every fill leaves you holding inventory that
    can lose money if the price moves. A-S solves for the quotes that maximise
    a risk-averse (exponential) utility of final wealth, and the answer comes
    in two clean pieces:

      reservation price  r = s - q * gamma * sigma^2 * (T - t)
          Where you are indifferent to one more share. Long inventory (q > 0)
          pushes r BELOW the mid, so your quotes shade down: your ask becomes
          more attractive (shedding stock) and your bid less so.

      optimal spread     delta = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)
          The first term is a risk premium (more volatility, more remaining
          time, wider). The second comes from the order-arrival model: fills
          arrive at rate A*exp(-k * distance), so k says how fast fill
          probability decays as you quote further from the mid.

    gamma is risk aversion. gamma -> 0 recovers a symmetric maker that ignores
    inventory; larger gamma means stronger skew and tighter inventory control.

    Units: everything is in TICKS and STEPS. sigma is the std of the mid's
    per-step change in ticks, (T - t) is in steps, k is per tick, q is shares.

    Horizon choice: ROLLING, (T - t) = `horizon` steps, constant. The paper has
    a terminal time (end of the trading day), so risk aversion fades to zero as
    the close approaches and is huge at the open. This simulation has no close,
    and a terminal clock would make behaviour depend on how many steps you
    happened to run. A rolling horizon ("I care about the next H steps of
    risk") is the common practical variant and keeps the maker stationary.
    """

    def __init__(self, name, gamma=0.1, k=1.5, horizon=5, window=50,
                 sigma_floor=0.25, max_offset=50, size=10, max_inv=200, **kw):
        super().__init__(name, spread=0, size=size, max_inv=max_inv, **kw)
        self.max_offset = max_offset
        self.gamma, self.k, self.horizon = gamma, k, horizon
        self.window, self.sigma_floor = window, sigma_floor
        self.last_quote = (None, None)   # (bid, ask) ticks, for charts

    def sigma(self, history):
        """Per-step std of mid changes over the recent window, in ticks.

        Floored: early on (or in a dead-calm patch) the estimate is ~0, and a
        zero sigma would make the maker ignore inventory risk entirely.
        """
        h = history[-(self.window + 1):]
        if len(h) < 3:
            return self.sigma_floor
        d = [b - a for a, b in zip(h, h[1:])]
        m = sum(d) / len(d)
        var = sum((x - m) ** 2 for x in d) / (len(d) - 1)
        return max(math.sqrt(var), self.sigma_floor)

    def quotes(self, mid, history):
        """(reservation price, total spread) in ticks, as floats."""
        risk = self.gamma * self.sigma(history) ** 2 * self.horizon
        r = mid - self.position * risk
        delta = risk + (2 / self.gamma) * math.log(1 + self.gamma / self.k)
        return r, delta

    def act(self, book, mid, history, rnd):
        # Fair value from the book as it stands NOW, after our own quotes were
        # cancelled. The `mid` handed to every agent is measured before the
        # cancel, so it includes our previous quotes. A maker that sits at the
        # top of book and centres on that mid is centring on itself: in
        # testing, that froze the price completely (mid constant for 1000
        # steps, inventory PnL exactly zero). Excluding our own quotes is what
        # a real maker does too: you estimate fair value from everyone else.
        bb = ba = None
        if book is not None:
            bb, ba = book.best_bid(), book.best_ask()
        s = (bb + ba) / 2 if bb is not None and ba is not None else mid
        r, delta = self.quotes(s, history)
        bid = math.floor(r - delta / 2)    # round AWAY from the mid: never
        ask = math.ceil(r + delta / 2)     # quote tighter than the model says
        # Sanity cap: never rest further than max_offset ticks from fair value.
        # With high gamma the formula can ask for quotes $5+ away; when one
        # side of the book is empty, the next market order sweeps straight
        # into that quote at an absurd price (a mini flash crash). Real desks
        # cap this too.
        bid = max(bid, math.floor(s) - self.max_offset)
        ask = min(ask, math.ceil(s) + self.max_offset)
        # Stay passive. A bid at or above the best ask would execute
        # immediately, turning a liquidity-providing quote into a liquidity-
        # taking trade that pays the spread instead of earning it.
        # If a side is empty, fall back to the step's mid as the reference:
        # with no other asks, an unclamped long-inventory ask can land far
        # below the market (even below zero), and the first market buy lifts
        # it. In testing that one fill lost $200k on a 10-share position.
        hi = ba if ba is not None else mid + 1
        lo = bb if bb is not None else mid - 1
        bid = max(1, min(bid, hi - 1))
        ask = max(ask, lo + 1, bid + 1)
        self.last_quote = (bid, ask)
        orders = []
        if self.position < self.max_inv:          # hard inventory limit
            orders.append(("limit", BUY, bid, self.size))
        if self.position > -self.max_inv:
            orders.append(("limit", SELL, ask, self.size))
        return orders


class Momentum(Agent):
    def __init__(self, name, lookback=10, size=5, **kw):
        super().__init__(name, **kw)
        self.lookback, self.size = lookback, size

    def act(self, book, mid, history, rnd):
        if len(history) <= self.lookback:
            return []
        change = history[-1] - history[-self.lookback]
        if change > 0 and self.position < 100:
            return [("market", BUY, None, self.size)]
        if change < 0 and self.position > -100:
            return [("market", SELL, None, self.size)]
        return []


class MeanReverter(Agent):
    def __init__(self, name, lookback=30, size=5, band=8, **kw):
        super().__init__(name, **kw)
        self.lookback, self.size, self.band = lookback, size, band

    def act(self, book, mid, history, rnd):
        if len(history) < self.lookback:
            return []
        avg = sum(history[-self.lookback:]) / self.lookback
        if mid < avg - self.band and self.position < 100:
            return [("market", BUY, None, self.size)]
        if mid > avg + self.band and self.position > -100:
            return [("market", SELL, None, self.size)]
        return []


class Noise(Agent):
    def __init__(self, name, size=3, spread=15, rate=0.5, **kw):
        super().__init__(name, **kw)
        self.size, self.spread, self.rate = size, spread, rate

    def act(self, book, mid, history, rnd):
        if rnd.random() > self.rate:
            return []
        side = rnd.choice([BUY, SELL])
        if rnd.random() < 0.3:
            return [("market", side, None, self.size)]
        off = rnd.randint(1, self.spread)
        price = mid - off if side == BUY else mid + off
        return [("limit", side, price, self.size)]


def default_agents():
    return [
        MarketMaker("Quill", spread=4, size=12),
        MarketMaker("Vane", spread=6, size=8),
        Momentum("Chase", lookback=8, size=6),
        MeanReverter("Anchor", lookback=25, size=6),
        Noise("Static", size=4),
        Noise("Drift", size=3),
    ]


class Simulation:
    def __init__(self, agents=None, seed=7, start=START_PRICE):
        self.book = Book()
        self.agents = agents if agents is not None else default_agents()
        self.rnd = random.Random(seed)
        self.owner: dict[int, tuple] = {}   # order id -> (agent, side)
        self.last = start
        self.history = [start]
        self.bars = []
        self._seen_events = 0
        self._start_cash = sum(a.cash for a in self.agents)

    # ---- helpers -------------------------------------------------------
    def mid(self):
        bb, ba = self.book.best_bid(), self.book.best_ask()
        if bb is not None and ba is not None:
            return (bb + ba) // 2
        return self.last

    def _submit(self, agent, kind, side, price, qty):
        if kind == "market":
            oid = self.book.market(side, qty)
        else:
            oid = self.book.limit(side, price, qty)
        self.owner[oid] = (agent, side)
        if kind == "limit":
            agent.live.append(oid)

    def _settle(self):
        """Attribute every new trade event to both sides."""
        prices = []
        for ev in self.book.events_since(self._seen_events):
            if ev[0] != "trade":
                continue
            # Index rather than unpack: events carry trailing seq/ts fields.
            price, qty, aggressor_id, resting_id = ev[1], ev[2], ev[3], ev[4]
            for oid in (aggressor_id, resting_id):
                who = self.owner.get(oid)
                if who:
                    who[0].on_fill(who[1], price, qty)
            prices.append(price)
        self._seen_events = self.book.event_count()
        return prices

    def _check(self):
        """Enforce the closed-economy invariants.

        These raise instead of assert on purpose: `python -O` strips assert
        statements, so an optimized deployment would run with every safety
        check silently removed. An invariant you only check in debug builds is
        not an invariant.
        """
        pos = sum(a.position for a in self.agents)
        if pos != 0:
            raise RuntimeError(f"shares created from nothing: net position {pos}")
        cash = sum(a.cash for a in self.agents)
        if abs(cash - self._start_cash) >= 1e-6:
            raise RuntimeError(f"money created: {cash:.4f} != {self._start_cash:.4f}")
        bb, ba = self.book.best_bid(), self.book.best_ask()
        if bb is not None and ba is not None and bb >= ba:
            raise RuntimeError(f"crossed book {bb}/{ba}")

    # ---- main loop -----------------------------------------------------
    def step(self):
        mid = self.mid()
        # Makers cancel and requote each step; otherwise stale quotes pile up
        # and the book fills with orders nobody intends to honour.
        makers = [a for a in self.agents if isinstance(a, MarketMaker)]
        for a in makers:
            for oid in a.live:
                self.book.cancel(oid)
            a.live.clear()
            a.mark = mid      # the mid every fill this step is measured against

        order = list(self.agents)
        self.rnd.shuffle(order)          # no agent gets a permanent time edge
        for a in order:
            for kind, side, price, qty in a.act(self.book, mid, self.history, self.rnd):
                self._submit(a, kind, side, price, qty)

        traded = self._settle()

        # Drop attribution entries for orders that can no longer trade. Without
        # this, `owner` grows by every order ever submitted for the life of the
        # run, and non-maker agents accumulate dead ids in `live` forever.
        if len(self.owner) > 20_000:
            alive = set(self.book.order_ids())
            self.owner = {oid: v for oid, v in self.owner.items() if oid in alive}
            for a in self.agents:
                a.live = [oid for oid in a.live if oid in alive]
        if traded:
            self.last = traded[-1]
            self.bars.append({
                "o": traded[0] * TICK, "h": max(traded) * TICK,
                "l": min(traded) * TICK, "c": traded[-1] * TICK,
                "v": len(traded),
            })
        new_mid = self.mid()
        self.history.append(new_mid)
        # Inventory PnL: the position held after this step's fills, times the
        # mid change from this step's mark to the next step's mark. Why this
        # pairing makes the identity exact (q = position, m = mark):
        #   q_end*m_end = sum_s [ q_{s+1}*(m_{s+1}-m_s) + (q_{s+1}-q_s)*m_s ]
        # The first sum is inventory PnL; the second, plus the cash paid for
        # the fills, is exactly spread capture measured at m_s. So
        #   equity(m_end) - start_cash == spread capture + inventory PnL.
        for a in makers:
            a.inv_ticks += a.position * (new_mid - mid)
        self._check()

    def run(self, steps=500):
        for _ in range(steps):
            self.step()
        return self.result()

    def result(self):
        mark = self.mid()
        return {
            "bars": self.bars,
            "last_price": round(self.last * TICK, 2),
            "trades": sum(1 for e in self.book.events_since(0) if e[0] == "trade"),
            "agents": sorted(
                [{"name": a.name, "kind": type(a).__name__,
                  "position": a.position,
                  "cash": round(a.cash, 2),
                  "equity": round(a.equity(mark), 2),
                  "pnl": round(a.equity(mark) - a.start_cash, 2)}
                 for a in self.agents],
                key=lambda d: -d["pnl"]),
        }


def _maker_summary(m, inv, mid_end):
    total = m.equity(mid_end) - m.start_cash
    parts = m.pnl_parts()
    n = len(inv)
    mean = sum(inv) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in inv) / n)
    return {
        "total_pnl": round(total, 2),
        "spread_capture": round(parts["spread"], 2),
        "inventory_pnl": round(parts["inventory"], 2),
        "identity_error": abs(total - parts["spread"] - parts["inventory"]),
        "inventory_std": round(std, 2),
        "max_abs_inventory": max(abs(x) for x in inv),
        "fills": m.fills,
        "pnl_per_fill": round(total / m.fills, 4) if m.fills else 0.0,
        "time_at_limit": round(sum(abs(x) >= m.max_inv for x in inv) / n, 4),
    }


def run_maker(gamma=0.1, k=1.5, steps=1000, seed=7, inventory_limit=200,
              horizon=5, size=10, naive_spread=4, max_points=600):
    """A-S maker vs the naive MarketMaker, same seed, same background market.

    The subject maker is added as a seventh agent alongside default_agents().
    Note the two markets share a seed but not a price path: the subject's own
    quotes change what trades, so the paths diverge after the first fills.
    That is the honest comparison (your quotes move the market); a fixed
    replayed path would hide it.
    """
    if gamma <= 0 or k <= 0:
        raise ValueError("gamma and k must be positive")
    steps = int(steps)
    if not 10 <= steps <= 20_000:
        raise ValueError("steps must be in [10, 20000]")

    def one(subject):
        sim = Simulation(agents=default_agents() + [subject], seed=seed)
        rows = []
        for _ in range(steps):
            sim.step()
            bid, ask = getattr(subject, "last_quote", (None, None))
            rows.append((sim.history[-1], subject.position, bid, ask,
                         subject.spread_ticks, subject.inv_ticks,
                         subject.equity(sim.history[-1]) - subject.start_cash))
        return sim, rows

    a_s = AvellanedaStoikov("A-S", gamma=gamma, k=k, horizon=horizon,
                            size=size, max_inv=inventory_limit)
    naive = MarketMaker("Naive", spread=naive_spread, size=size, max_inv=inventory_limit)
    sim_a, rows_a = one(a_s)
    sim_n, rows_n = one(naive)

    stride = max(1, -(-steps // max_points))
    pick = rows_a[::stride]
    pick_n = rows_n[::stride]
    px = lambda t: None if t is None else round(t * TICK, 2)
    return {
        "params": {"gamma": gamma, "k": k, "steps": steps, "seed": seed,
                   "inventory_limit": inventory_limit, "horizon": horizon,
                   "size": size, "naive_spread": naive_spread},
        "step": list(range(1, steps + 1, stride)),
        "paths": {
            "mid": [px(r[0]) for r in pick],
            "inventory": [r[1] for r in pick],
            "bid": [px(r[2]) for r in pick],
            "ask": [px(r[3]) for r in pick],
            "spread_capture": [round(r[4] * TICK, 2) for r in pick],
            "inventory_pnl": [round(r[5] * TICK, 2) for r in pick],
            "total_pnl": [round(r[6], 2) for r in pick],
            "naive_mid": [px(r[0]) for r in pick_n],
            "naive_inventory": [r[1] for r in pick_n],
            "naive_total_pnl": [round(r[6], 2) for r in pick_n],
        },
        "summary": {
            "avellaneda_stoikov": _maker_summary(a_s, [r[1] for r in rows_a], sim_a.history[-1]),
            "naive": _maker_summary(naive, [r[1] for r in rows_n], sim_n.history[-1]),
        },
    }


def demo():
    sim = Simulation(seed=7)
    out = sim.run(400)

    assert out["trades"] > 100, f"market barely traded: {out['trades']}"
    assert len(out["bars"]) > 50, "not enough bars produced"
    assert sum(a["position"] for a in out["agents"]) == 0, "positions must net to zero"

    # PnL is zero-sum at the marking price: one agent's gain is another's loss.
    total_pnl = sum(a["pnl"] for a in out["agents"])
    assert abs(total_pnl) < 1e-6, f"PnL must be zero-sum, got {total_pnl}"

    # bars must be internally consistent
    for b in out["bars"]:
        assert b["l"] <= b["o"] <= b["h"], f"bad bar {b}"
        assert b["l"] <= b["c"] <= b["h"], f"bad bar {b}"
        assert b["v"] > 0

    # price actually moved (a market that never moves proves nothing)
    closes = [b["c"] for b in out["bars"]]
    assert max(closes) > min(closes), "price never moved"

    # determinism: same seed, same market
    assert Simulation(seed=7).run(120)["bars"] == Simulation(seed=7).run(120)["bars"], \
        "same seed must reproduce the market exactly"
    assert Simulation(seed=8).run(120)["bars"] != Simulation(seed=7).run(120)["bars"], \
        "different seeds should diverge"

    # a market with no maker should barely trade: liquidity has to come from somewhere
    dry = Simulation(agents=[Noise("a"), Noise("b")], seed=3).run(200)
    assert dry["trades"] < out["trades"], "makers should increase traded volume"

    # PnL must be measured against each agent's OWN starting cash, so agents
    # funded differently still net to zero
    mixed = Simulation(agents=[MarketMaker("rich", cash=500_000),
                               Noise("poor", cash=1_000),
                               Momentum("mid", cash=50_000)], seed=5).run(150)
    assert abs(sum(a["pnl"] for a in mixed["agents"])) < 1e-6, \
        "zero-sum broken when agents start with different cash"

    # inventory skew: a long maker must quote lower on BOTH sides than a flat one
    def quotes(pos):
        mm = MarketMaker("m", spread=4, max_inv=200)
        mm.position = pos
        return {side: price for _, side, price, _ in mm.act(None, 10_000, [], random.Random(0))}

    flat, long_, short_ = quotes(0), quotes(100), quotes(-100)
    assert long_[BUY] < flat[BUY] and long_[SELL] < flat[SELL], \
        "long inventory must push both quotes down to attract sellers"
    assert short_[BUY] > flat[BUY] and short_[SELL] > flat[SELL], \
        "short inventory must push both quotes up"

    # at max inventory the maker must stop adding to it
    assert BUY not in quotes(200), "maker must stop bidding when fully long"
    assert SELL not in quotes(-200), "maker must stop offering when fully short"

    # ---- Avellaneda-Stoikov ----
    def as_quotes(pos, gamma=0.1):
        m = AvellanedaStoikov("as", gamma=gamma)
        m.position = pos
        hist = [10_000 + (i % 3) for i in range(60)]          # some volatility
        return {side: p for _, side, p, _ in m.act(None, 10_000, hist, None)}

    f0, fl, fs = as_quotes(0), as_quotes(10), as_quotes(-10)
    assert f0[BUY] < 10_000 < f0[SELL], "flat A-S maker quotes around the mid"
    assert fl[BUY] < f0[BUY] and fl[SELL] < f0[SELL], "long inventory lowers r"
    assert fs[BUY] > f0[BUY] and fs[SELL] > f0[SELL], "short inventory raises r"
    assert as_quotes(10, gamma=0.3)[BUY] < fl[BUY], "more risk aversion, more skew"
    assert BUY not in as_quotes(200), "hard inventory limit"

    # passivity: even a huge skew must not cross the live book
    b = Book()
    b.limit(BUY, 9_990, 5)
    b.limit(SELL, 9_995, 5)
    m = AvellanedaStoikov("as", gamma=1.0, max_offset=10_000)
    m.position = 150
    q = {side: p for _, side, p, _ in m.act(b, 9_992, [10_000, 10_005] * 30, None)}
    assert q[BUY] < 9_995 and q[SELL] > 9_990, f"quote went aggressive: {q}"

    t0 = time.perf_counter()
    rm = run_maker()
    t_maker = time.perf_counter() - t0
    a_s, nv = rm["summary"]["avellaneda_stoikov"], rm["summary"]["naive"]
    for name, sm in (("A-S", a_s), ("naive", nv)):
        assert sm["identity_error"] < 1e-6, f"{name}: total != spread + inventory"
        assert sm["fills"] > 50, f"{name} barely traded"
    # The paper's core claim: skewing by inventory keeps inventory tighter.
    assert a_s["inventory_std"] < nv["inventory_std"], (a_s, nv)
    assert run_maker(steps=300) == run_maker(steps=300), "run_maker must be deterministic"
    assert len(rm["paths"]["mid"]) == len(rm["step"]) <= 600
    big = run_maker(steps=2_000)["paths"]["mid"]
    assert len(big) <= 600, "paths must be downsampled"

    print(f"ok — {out['trades']} trades, {len(out['bars'])} bars, "
          f"last {out['last_price']}, zero-sum PnL, deterministic")
    print(f"ok — A-S vs naive ({t_maker:.2f}s): "
          f"PnL {a_s['total_pnl']} vs {nv['total_pnl']}, "
          f"spread {a_s['spread_capture']} vs {nv['spread_capture']}, "
          f"inventory {a_s['inventory_pnl']} vs {nv['inventory_pnl']}, "
          f"inv std {a_s['inventory_std']} vs {nv['inventory_std']}, "
          f"fills {a_s['fills']} vs {nv['fills']}; identity holds for both")


if __name__ == "__main__":
    demo()
