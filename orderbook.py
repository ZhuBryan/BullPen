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

Run `python orderbook.py` to execute the self-check.
"""

from collections import deque
from dataclasses import dataclass

BUY, SELL = "B", "S"


@dataclass
class Order:
    id: int
    side: str
    price: int | None  # None = market order
    qty: int
    remaining: int


class Book:
    def __init__(self):
        self.bids: dict[int, deque[Order]] = {}  # price -> FIFO queue
        self.asks: dict[int, deque[Order]] = {}
        self.orders: dict[int, Order] = {}       # id -> order, for O(1) cancel lookup
        self.events: list[tuple] = []
        self._next_id = 1

    # ---- introspection -------------------------------------------------
    # ponytail: max()/min() is O(levels) per call. Fine to thousands of levels;
    # swap for a heap or flat tick-indexed array when benchmarking says so.
    def best_bid(self):
        return max(self.bids) if self.bids else None

    def best_ask(self):
        return min(self.asks) if self.asks else None

    def depth(self, side, levels=5):
        book = self.bids if side == BUY else self.asks
        prices = sorted(book, reverse=(side == BUY))[:levels]
        return [(p, sum(o.remaining for o in book[p])) for p in prices]

    # ---- commands ------------------------------------------------------
    def limit(self, side, price, qty):
        return self._submit(side, int(price), qty)

    def market(self, side, qty):
        return self._submit(side, None, qty)

    def cancel(self, order_id):
        o = self.orders.pop(order_id, None)
        if o is None:
            self.events.append(("reject_cancel", order_id))
            return False
        book = self.bids if o.side == BUY else self.asks
        level = book.get(o.price)
        if level:
            # ponytail: O(level size) removal. Intrusive linked list is the
            # upgrade if cancel-heavy workloads show up in the profile.
            try:
                level.remove(o)
            except ValueError:
                pass
            if not level:
                del book[o.price]
        self.events.append(("canceled", order_id, o.remaining))
        return True

    # ---- engine --------------------------------------------------------
    def _submit(self, side, price, qty):
        if side not in (BUY, SELL) or qty <= 0:
            raise ValueError("bad order")
        o = Order(self._next_id, side, price, int(qty), int(qty))
        self._next_id += 1
        self.events.append(("accepted", o.id, side, price, qty))
        self._match(o)
        if o.remaining:
            if price is None:
                self.events.append(("canceled", o.id, o.remaining))  # unfilled market qty
                o.remaining = 0
            else:
                book = self.bids if side == BUY else self.asks
                book.setdefault(price, deque()).append(o)
                self.orders[o.id] = o
        return o.id

    def _crosses(self, o, best):
        if best is None:
            return False
        if o.price is None:           # market order takes any price
            return True
        return o.price >= best if o.side == BUY else o.price <= best

    def _match(self, o):
        opp = self.asks if o.side == BUY else self.bids
        while o.remaining:
            best = self.best_ask() if o.side == BUY else self.best_bid()
            if not self._crosses(o, best):
                break
            level = opp[best]
            while o.remaining and level:
                resting = level[0]
                traded = min(o.remaining, resting.remaining)
                o.remaining -= traded
                resting.remaining -= traded
                self.events.append(("trade", best, traded, o.id, resting.id))
                if resting.remaining == 0:
                    level.popleft()
                    self.orders.pop(resting.id, None)
            if not level:
                del opp[best]


# ---------------------------------------------------------------------------
# Self-check: the invariants that must hold after every event.
# ---------------------------------------------------------------------------
def _check_invariants(b):
    bb, ba = b.best_bid(), b.best_ask()
    assert bb is None or ba is None or bb < ba, f"crossed book: {bb} >= {ba}"
    for book in (b.bids, b.asks):
        for price, level in book.items():
            assert level, f"empty level left at {price}"
            for o in level:
                assert o.remaining > 0, "fully filled order still resting"
                assert b.orders.get(o.id) is o, "resting order missing from index"


def demo():
    # exact fill
    b = Book()
    b.limit(SELL, 100, 10)
    b.limit(BUY, 100, 10)
    assert [e for e in b.events if e[0] == "trade"] == [("trade", 100, 10, 2, 1)]
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
