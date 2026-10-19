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

_BAD_OP = ("each op must be a tuple ('L', side, price, qty), "
           "('M', side, qty) or ('X', order_id)")


# Default clock: monotonic, so it cannot run backwards mid-session. The C
# function itself rather than a Python wrapper around it: the clock is read
# once per event, and a wrapper frame cost more than the read (profiled).
monotonic_ns = time.perf_counter_ns


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


@dataclass(slots=True)   # slots: faster attribute access, less memory per order
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
        # side -> (queues, sorted prices, live counts), built once: the hot
        # paths look a side up on every order, and building that tuple per
        # call showed up in the profile.
        self._side = {BUY: (self.bids, self._bid_prices, self._bid_live),
                      SELL: (self.asks, self._ask_prices, self._ask_live)}

    # ---- level bookkeeping ---------------------------------------------
    def _levels(self, side):
        return self._side[BUY if side == BUY else SELL]

    # ---- introspection -------------------------------------------------
    def best_bid(self):
        return self._bid_prices[-1] if self._bid_prices else None

    def best_ask(self):
        return self._ask_prices[0] if self._ask_prices else None

    def depth(self, side, levels=5):
        book, prices, _ = self._levels(side)
        if side != BUY:
            chosen = prices[:levels]
        elif levels is not None and levels > 0:
            chosen = prices[-levels:][::-1]   # copy only the levels asked for
        else:
            chosen = prices[::-1][:levels]    # same slice as always, odd inputs included
        out = []
        for p in chosen:
            q = 0
            for o in book[p]:                 # skip husks of lazy cancels
                if not o.cancelled:
                    q += o.remaining
            out.append((p, q))
        return out

    # ---- portable interface --------------------------------------------
    # The subset of Book that the Rust port also exposes through PyO3 (see
    # SPEC.md section 7). Code that should run on either engine (the market
    # simulator, the order-book backtester) uses only these plus limit /
    # market / cancel / submit_many / best_bid / best_ask / depth, never `events` or
    # `orders` directly: handing a whole Rust-side list to Python on every
    # step would turn an O(n) simulation into O(n^2) copying.
    def event_count(self):
        return len(self.events)

    def events_since(self, n):
        return self.events[n:]

    def has_order(self, order_id):
        return order_id in self.orders

    def order_ids(self):
        return list(self.orders)

    def submit_many(self, ops):
        """Apply a list of commands in order; return one result per command.

            ("L", side, price, qty)  -> limit(side, price, qty)  -> order id
            ("M", side, qty)         -> market(side, qty)        -> order id
            ("X", order_id)          -> cancel(order_id)         -> bool

        Exactly equivalent to making the single calls one after another: same
        events, same ids, same validation, same exceptions. That includes what
        happens on bad input: the bad command raises the same ValueError its
        single call would, the commands before it STAY APPLIED (they already
        happened, as they would have with separate calls), and the ones after
        it are not run. A malformed command (unknown tag, wrong length, not a
        tuple) raises ValueError at its position in the same way.

        Why it exists: on the Rust engine every call from Python pays a fixed
        cost to cross the language boundary and convert its arguments, however
        little work the call does. A market maker requoting two orders, or
        cancelling ten, is one crossing instead of two or ten. Here in Python
        it is just a loop, kept so both engines share one interface.
        """
        out = []
        for op in ops:
            n = len(op) if isinstance(op, tuple) else 0
            tag = op[0] if n else None
            if tag == "L" and n == 4:
                out.append(self.limit(op[1], op[2], op[3]))
            elif tag == "M" and n == 3:
                out.append(self.market(op[1], op[2]))
            elif tag == "X" and n == 2:
                out.append(self.cancel(op[1]))
            else:
                raise ValueError(_BAD_OP)
        return out

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
        book, prices, live = self._side[o.side]
        p = o.price
        n = live.get(p)
        if n is not None:
            if n > 1:
                live[p] = n - 1
            else:                     # last live order: the level goes too
                book.pop(p, None)
                del live[p]
                i = bisect_left(prices, p)
                if i < len(prices) and prices[i] == p:
                    prices.pop(i)
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
        # A float is refused even when it is whole (100.0): it means the caller
        # did arithmetic in floating point somewhere upstream, which is how
        # money drifts. This is also exactly what the Rust engine accepts, so
        # the two can't disagree on input (engine/test_pyo3.py checks it).
        # (`type(x) is int` is only a fast path for the common case: a plain
        # int passes every check below except the sign, so skip straight to it.)
        if type(qty) is not int and (isinstance(qty, float) or int(qty) != qty):
            raise ValueError("qty must be a whole number of shares")
        if price is not None:
            if type(price) is not int:
                if isinstance(price, float) or int(price) != price:
                    raise ValueError("price must be a positive whole number of ticks")
                price = int(price)
            if price <= 0:
                raise ValueError("price must be a positive whole number of ticks")
        q = int(qty)
        o = Order(self._next_id, side, price, q, q)
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
        book, prices, live = self._side[o.side]
        p = o.price
        level = book.get(p)
        if level is None:
            book[p] = deque((o,))
            live[p] = 1
            insort(prices, p)
        else:
            level.append(o)
            live[p] += 1
        self.orders[o.id] = o

    def _match(self, o):
        # Sweep the opposite side while the order crosses: a buy crosses an
        # ask at or below its limit, a sell a bid at or above it, and a
        # market order (no limit) crosses anything. Written out inline, with
        # no helper calls, because this runs for every incoming order and
        # most of them cross nothing.
        buy = o.side == BUY
        opp, prices, live = self._side[SELL if buy else BUY]
        limit = o.price                   # None = market order, takes any price
        events, clock, orders = self.events, self._clock, self.orders
        while o.remaining and prices:
            best = prices[0] if buy else prices[-1]
            if limit is not None and (limit < best if buy else limit > best):
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
                events.append(("trade", best, traded, o.id, resting.id, self._seq, clock()))
                if resting.remaining == 0:
                    level.popleft()
                    live[best] -= 1
                    orders.pop(resting.id, None)
            if live.get(best, 0) <= 0:
                # The level is used up. It is the best one, so it sits at the
                # end of the sorted price list nearest the spread.
                del opp[best]
                live.pop(best, None)
                prices.pop(0 if buy else -1)


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
