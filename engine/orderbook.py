"""
Limit order book matching engine — price-time priority.

Prices are INTEGER TICKS, never floats. 0.1 + 0.2 != 0.3, and money bugs from
float rounding are silent and unrecoverable. Convert at the edges only:
dollars = ticks * tick_size.

Matching rule (what real exchanges do):
  1. Better price wins.
  2. At the same price, whoever arrived first wins (FIFO).
An incoming order sweeps the opposite side while prices cross, filling at the
RESTING order's price (the resting order set the terms), and rests whatever
remains. Market orders never rest — the remainder is cancelled.

Two data-structure decisions, both made because the profiler said so
(see bench.py, and the numbers in README.md):

  * SORTED PRICE INDEX. Keeping only dicts of price -> queue forces an
    interpreted max()/min() scan on every best-bid/ask lookup. bisect keeps a
    sorted list alongside; insort/pop move memory in C, which beats a Python
    loop by enough to matter even though both are nominally O(levels).

  * LAZY CANCELLATION. Removing an order from the middle of its queue is an
    O(queue) scan, and deque random access makes it worse. Profiling showed
    cancels at a 82us median against 2.4us for limits — 34x, on 15% of the
    traffic. So cancels now just flag the order and decrement a live counter;
    the flagged husk is skipped when matching reaches it. This is what real
    exchanges do, and it is why a per-level live count exists: without it a
    level holding nothing but cancelled husks would still advertise a price.

Run `python orderbook.py` to execute the self-check.
"""

import time
from bisect import bisect_left, insort
from collections import deque
from dataclasses import dataclass

BUY, SELL = "B", "S"


def monotonic_ns():
    """Default clock. Monotonic so it cannot run backwards mid-session."""
    return time.perf_counter_ns()


def counter_clock():
    """Deterministic stand-in clock, for tests and cross-engine comparison.

    Wall-clock timestamps are process-local by definition: two engines replaying
    the same stream will never produce the same ones. Injecting a counter makes
    a log reproducible when that is what matters.
    """
    n = [0]

    def tick():
        n[0] += 1
        return n[0]

    return tick


@dataclass
class Order:
    id: int
    side: str
    price: int | None  # None = market order
    qty: int
    remaining: int
    cancelled: bool = False


class Book:
    def __init__(self, clock=None):
        # Every event carries a monotonic sequence number and a timestamp, the
        # way a real venue's feed does (ITCH stamps every message). `seq` is the
        # authoritative ordering: it is deterministic and survives replay, while
        # a wall clock does not. `clock` is injectable so a test or a
        # cross-engine comparison can make the log reproducible.
        self._clock = clock or monotonic_ns
        self._seq = 0
        self.bids: dict[int, deque[Order]] = {}  # price -> FIFO queue
        self.asks: dict[int, deque[Order]] = {}
        self.orders: dict[int, Order] = {}       # id -> order, O(1) cancel lookup
        self.events: list[tuple] = []
        self._next_id = 1
        self._bid_prices: list[int] = []         # sorted ascending
        self._ask_prices: list[int] = []
        self._bid_live: dict[int, int] = {}      # price -> live (uncancelled) orders
        self._ask_live: dict[int, int] = {}

    # ---- level bookkeeping ---------------------------------------------
    def _levels(self, side):
        if side == BUY:
            return self.bids, self._bid_prices, self._bid_live
        return self.asks, self._ask_prices, self._ask_live

    def _del_level(self, side, price):
        book, prices, live = self._levels(side)
        book.pop(price, None)
        live.pop(price, None)
        i = bisect_left(prices, price)
        if i < len(prices) and prices[i] == price:
            prices.pop(i)

    # ---- introspection -------------------------------------------------
    def best_bid(self):
        return self._bid_prices[-1] if self._bid_prices else None

    def best_ask(self):
        return self._ask_prices[0] if self._ask_prices else None

    def depth(self, side, levels=5):
        book, prices, _ = self._levels(side)
        chosen = prices[::-1][:levels] if side == BUY else prices[:levels]
        return [(p, sum(o.remaining for o in book[p] if not o.cancelled))
                for p in chosen]

    # ---- commands ------------------------------------------------------
    def limit(self, side, price, qty):
        # Pass price through unconverted so _submit can reject a fractional one.
        # Converting here would hide the very thing the check is looking for.
        return self._submit(side, price, qty)

    def market(self, side, qty):
        return self._submit(side, None, qty)

    def cancel(self, order_id):
        o = self.orders.pop(order_id, None)
        if o is None:
            self._seq += 1
            self.events.append(("reject_cancel", order_id, self._seq, self._clock()))
            return False
        o.cancelled = True
        _, _, live = self._levels(o.side)
        if o.price in live:
            live[o.price] -= 1
            if live[o.price] <= 0:
                self._del_level(o.side, o.price)
        self._seq += 1
        self.events.append(("canceled", order_id, o.remaining, self._seq, self._clock()))
        return True

    # ---- engine --------------------------------------------------------
    def _submit(self, side, price, qty):
        if side not in (BUY, SELL) or qty <= 0:
            raise ValueError("bad order")
        # Quantities and prices are integers (shares, ticks). Silently calling
        # int() here would turn a 0.5 order into 0: accepted, never rested,
        # never cancelled, and therefore missing from the shares ledger. Reject
        # loudly instead, and make the caller choose a lot size.
        if int(qty) != qty:
            raise ValueError("qty must be a whole number of shares")
        if price is not None and int(price) != price:
            raise ValueError("price must be a whole number of ticks")
        price = int(price) if price is not None else None
        o = Order(self._next_id, side, price, int(qty), int(qty))
        self._next_id += 1
        self._seq += 1
        self.events.append(("accepted", o.id, side, price, qty, self._seq, self._clock()))
        self._match(o)
        if o.remaining:
            if price is None:
                self._seq += 1
                self.events.append(("canceled", o.id, o.remaining, self._seq, self._clock()))
                o.remaining = 0
            else:
                self._rest(o)
        return o.id

    def _rest(self, o):
        book, prices, live = self._levels(o.side)
        if o.price not in book:
            book[o.price] = deque()
            live[o.price] = 0
            insort(prices, o.price)
        book[o.price].append(o)
        live[o.price] += 1
        self.orders[o.id] = o

    def _crosses(self, o, best):
        if best is None:
            return False
        if o.price is None:           # market order takes any price
            return True
        return o.price >= best if o.side == BUY else o.price <= best

    def _match(self, o):
        opp_side = SELL if o.side == BUY else BUY
        opp, _, live = self._levels(opp_side)
        while o.remaining:
            best = self.best_ask() if o.side == BUY else self.best_bid()
            if not self._crosses(o, best):
                break
            level = opp[best]
            while o.remaining and level:
                resting = level[0]
                if resting.cancelled:     # husk left by a lazy cancel
                    level.popleft()
                    continue
                traded = min(o.remaining, resting.remaining)
                o.remaining -= traded
                resting.remaining -= traded
                self._seq += 1
                self.events.append(
                    ("trade", best, traded, o.id, resting.id, self._seq, self._clock()))
                if resting.remaining == 0:
                    level.popleft()
                    live[best] -= 1
                    self.orders.pop(resting.id, None)
            if live.get(best, 0) <= 0:
                self._del_level(opp_side, best)


# ---------------------------------------------------------------------------
# Self-check: the invariants that must hold after every event.
# ---------------------------------------------------------------------------
def _check_invariants(b):
    bb, ba = b.best_bid(), b.best_ask()
    assert bb is None or ba is None or bb < ba, f"crossed book: {bb} >= {ba}"
    for side in (BUY, SELL):
        book, prices, live = b._levels(side)
        assert prices == sorted(book), f"{side} price index out of sync"
        for price, level in book.items():
            actual = sum(1 for o in level if not o.cancelled)
            assert live[price] == actual, (
                f"live count wrong at {price}: {live[price]} != {actual}")
            assert actual > 0, f"level {price} advertised with no live orders"
            for o in level:
                if not o.cancelled:
                    assert o.remaining > 0, "fully filled order still resting"
                    assert b.orders.get(o.id) is o, "resting order missing from index"


def demo():
    # exact fill
    b = Book()
    b.limit(SELL, 100, 10)
    b.limit(BUY, 100, 10)
    assert [e[:5] for e in b.events if e[0] == "trade"] == [("trade", 100, 10, 2, 1)]
    assert b.best_bid() is None and b.best_ask() is None, "book should be empty"
    _check_invariants(b)

    # partial fill: remainder rests
    b = Book()
    b.limit(SELL, 100, 5)
    bid = b.limit(BUY, 100, 8)
    assert b.best_bid() == 100 and b.orders[bid].remaining == 3
    _check_invariants(b)

    # price-time priority: earlier order at the same price fills first
    b = Book()
    first = b.limit(SELL, 100, 5)
    second = b.limit(SELL, 100, 5)
    b.limit(BUY, 100, 5)
    trades = [e for e in b.events if e[0] == "trade"]
    assert trades[0][4] == first, "FIFO violated"
    assert b.orders[second].remaining == 5
    _check_invariants(b)

    # better price wins over earlier time
    b = Book()
    b.limit(SELL, 101, 5)
    cheap = b.limit(SELL, 100, 5)
    b.limit(BUY, 101, 5)
    assert [e for e in b.events if e[0] == "trade"][0][4] == cheap, "price priority violated"
    _check_invariants(b)

    # fills happen at the RESTING order's price, not the aggressor's
    b = Book()
    b.limit(SELL, 100, 5)
    b.limit(BUY, 105, 5)
    assert [e for e in b.events if e[0] == "trade"][0][1] == 100
    _check_invariants(b)

    # market order sweeps levels, unfilled remainder is cancelled not rested
    b = Book()
    b.limit(SELL, 100, 3)
    b.limit(SELL, 101, 3)
    b.market(BUY, 10)
    assert [e[2] for e in b.events if e[0] == "trade"] == [3, 3]
    assert b.best_bid() is None, "market order must not rest"
    _check_invariants(b)

    # cancel, including a half-filled order
    b = Book()
    oid = b.limit(SELL, 100, 10)
    b.limit(BUY, 100, 4)
    assert b.orders[oid].remaining == 6
    assert b.cancel(oid) is True
    assert b.best_ask() is None
    assert b.cancel(oid) is False, "double cancel must fail"
    _check_invariants(b)

    # lazy cancellation must not leave a phantom price level
    b = Book()
    a = b.limit(BUY, 100, 5)
    c = b.limit(BUY, 100, 5)
    b.cancel(a)
    assert b.best_bid() == 100, "level still has a live order"
    b.cancel(c)
    assert b.best_bid() is None, "phantom level: all orders cancelled"
    _check_invariants(b)

    # a cancelled husk must be skipped, not filled
    b = Book()
    dead = b.limit(SELL, 100, 5)
    alive = b.limit(SELL, 100, 5)
    b.cancel(dead)
    b.limit(BUY, 100, 5)
    trades = [e for e in b.events if e[0] == "trade"]
    assert len(trades) == 1 and trades[0][4] == alive, "filled a cancelled order"
    _check_invariants(b)

    # every event is stamped with a monotonic sequence number and a timestamp
    b = Book()
    b.limit(SELL, 100, 5)
    b.limit(BUY, 100, 5)
    oid = b.limit(BUY, 99, 5)
    b.cancel(oid)
    seqs = [e[-2] for e in b.events]
    assert seqs == list(range(1, len(seqs) + 1)), f"seq must be dense and monotonic: {seqs}"
    ts = [e[-1] for e in b.events]
    assert all(ts[i] <= ts[i + 1] for i in range(len(ts) - 1)), "timestamps went backwards"
    assert all(isinstance(t, int) for t in ts), "timestamps must be integer nanoseconds"

    # an injected clock makes the log reproducible, which a wall clock cannot be
    def run_with_counter():
        bk = Book(clock=counter_clock())
        bk.limit(SELL, 100, 5)
        bk.limit(BUY, 100, 5)
        return bk.events
    assert run_with_counter() == run_with_counter(), "injected clock must be deterministic"
    assert run_with_counter()[0][-1] == 1, "counter clock should start at 1"

    # fractional sizes must be refused, not silently truncated to zero
    b = Book()
    for bad in (0.5, 2.5):
        try:
            b.limit(BUY, 100, bad)
            raise AssertionError(f"qty {bad} was silently accepted")
        except ValueError:
            pass
    try:
        b.limit(BUY, 100.9, 5)
        raise AssertionError("fractional price was silently accepted")
    except ValueError:
        pass
    assert b.best_bid() is None and not b.orders, "rejected orders left residue"

    # re-resting at a price whose level was fully cancelled
    b = Book()
    x = b.limit(BUY, 100, 5)
    b.cancel(x)
    b.limit(BUY, 100, 7)
    assert b.best_bid() == 100 and b.depth(BUY)[0] == (100, 7)
    _check_invariants(b)

    # share conservation over a randomized stream
    import random
    rnd = random.Random(42)
    b, submitted = Book(), 0
    for _ in range(3000):
        side = rnd.choice([BUY, SELL])
        act = rnd.random()
        if act < 0.15 and b.orders:
            b.cancel(rnd.choice(list(b.orders)))
        elif act < 0.30:
            q = rnd.randint(1, 20)
            submitted += q
            b.market(side, q)
        else:
            q = rnd.randint(1, 20)
            submitted += q
            b.limit(side, rnd.randint(95, 105), q)
        _check_invariants(b)

    filled = sum(e[2] for e in b.events if e[0] == "trade")
    canceled = sum(e[2] for e in b.events if e[0] == "canceled")
    resting = sum(o.remaining for o in b.orders.values())
    # each fill consumes `traded` shares from BOTH the aggressor and the resting
    # order, so a trade of size q accounts for 2q of the submitted shares.
    assert 2 * filled + canceled + resting == submitted, (
        f"shares leaked: 2*{filled} + {canceled} + {resting} != {submitted}"
    )
    print(f"ok — {len(b.events)} events, {filled} filled, {canceled} canceled, {resting} resting")


if __name__ == "__main__":
    demo()
