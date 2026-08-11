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

import random

try:
    from engine.orderbook import BUY, SELL, Book
except ImportError:                        # running this file directly
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from engine.orderbook import BUY, SELL, Book

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
        for ev in self.book.events[self._seen_events:]:
            if ev[0] != "trade":
                continue
            _, price, qty, aggressor_id, resting_id = ev
            for oid in (aggressor_id, resting_id):
                who = self.owner.get(oid)
                if who:
                    who[0].on_fill(who[1], price, qty)
            prices.append(price)
        self._seen_events = len(self.book.events)
        return prices

    def _check(self):
        pos = sum(a.position for a in self.agents)
        assert pos == 0, f"shares created from nothing: net position {pos}"
        cash = sum(a.cash for a in self.agents)
        assert abs(cash - self._start_cash) < 1e-6, (
            f"money created: {cash:.4f} != {self._start_cash:.4f}")
        bb, ba = self.book.best_bid(), self.book.best_ask()
        assert bb is None or ba is None or bb < ba, f"crossed book {bb}/{ba}"

    # ---- main loop -----------------------------------------------------
    def step(self):
        mid = self.mid()
        # Makers cancel and requote each step; otherwise stale quotes pile up
        # and the book fills with orders nobody intends to honour.
        for a in self.agents:
            if isinstance(a, MarketMaker):
                for oid in a.live:
                    self.book.cancel(oid)
                a.live.clear()

        order = list(self.agents)
        self.rnd.shuffle(order)          # no agent gets a permanent time edge
        for a in order:
            for kind, side, price, qty in a.act(self.book, mid, self.history, self.rnd):
                self._submit(a, kind, side, price, qty)

        traded = self._settle()
        if traded:
            self.last = traded[-1]
            self.bars.append({
                "o": traded[0] * TICK, "h": max(traded) * TICK,
                "l": min(traded) * TICK, "c": traded[-1] * TICK,
                "v": len(traded),
            })
        self.history.append(self.mid())
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
            "trades": sum(1 for e in self.book.events if e[0] == "trade"),
            "agents": sorted(
                [{"name": a.name, "kind": type(a).__name__,
                  "position": a.position,
                  "cash": round(a.cash, 2),
                  "equity": round(a.equity(mark), 2),
                  "pnl": round(a.equity(mark) - a.start_cash, 2)}
                 for a in self.agents],
                key=lambda d: -d["pnl"]),
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

    print(f"ok — {out['trades']} trades, {len(out['bars'])} bars, "
          f"last {out['last_price']}, zero-sum PnL, deterministic")


if __name__ == "__main__":
    demo()
