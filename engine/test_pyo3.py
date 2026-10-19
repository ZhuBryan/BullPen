"""
Differential test: the Rust engine, called from Python through PyO3, must
behave exactly like the Python engine.

diff_test.py already compares the two engines through text files and the Rust
binary. This test compares them through the *binding*, which is the code path
the product actually uses, so it also catches mistakes in the translation layer:
a field in the wrong tuple position, a side string mangled, a None that came
back as 0.

Both books get the same random op stream (fixed seed, 20,000 ops: limits,
markets, and cancels, including cancels of ids that were never issued or are
already gone). After every op the new events are fetched with
events_since(), which is also how real callers consume the log, and compared
tuple for tuple except the timestamp (ts is process-local; seq is not and is
compared). Book state is compared at checkpoints.

Skips cleanly (exit 0) when the extension is not built:
    cd rust; maturin develop --release
"""

import random
import os
import sys

try:
    from engine.orderbook import Book as PyBook, counter_clock
except ImportError:
    from orderbook import Book as PyBook, counter_clock

N_OPS = 20_000
SEED = 7
CHECK_EVERY = 250


def gen_ops(n, seed):
    rnd = random.Random(seed)
    ops, issued = [], 0
    for _ in range(n):
        side = rnd.choice("BS")
        roll = rnd.random()
        if roll < 0.15 and issued:
            ops.append(("X", rnd.randint(1, issued)))        # live, filled or cancelled
        elif roll < 0.18:
            ops.append(("X", issued + rnd.randint(1, 50)))   # never issued
        elif roll < 0.30:
            ops.append(("M", side, rnd.randint(1, 60)))
            issued += 1
        else:
            off = rnd.randint(0, 10)                         # 0 = at the touch, crosses often
            price = 1_000 - off if side == "B" else 1_000 + off
            if rnd.random() < 0.2:                           # aggressive limit, sweeps levels
                price = 1_000 + off if side == "B" else 1_000 - off
            if rnd.random() < 0.02:
                # Rare extremes: 1 tick (a sell there sweeps every bid) and
                # prices near the top of what the integer types must hold, so
                # level ordering and overflow-adjacent values are exercised,
                # not only the 990-1010 band around the market.
                price = rnd.choice((1, 2, 10**9, 10**12, 2**53))
            ops.append(("L", side, price, rnd.randint(1, 60)))
            issued += 1
    return ops


def apply(book, op):
    if op[0] == "L":
        return book.limit(op[1], op[2], op[3])
    if op[0] == "M":
        return book.market(op[1], op[2])
    return book.cancel(op[1])


def state(book):
    return (book.best_bid(), book.best_ask(),
            book.depth("B"), book.depth("S"),
            book.depth("B", 1000), book.depth("S", 1000), book.depth("B", 1),
            book.event_count(), book.order_ids())


def run(rust_book_cls):
    py, rs = PyBook(clock=counter_clock()), rust_book_cls(deterministic=True)
    ops = gen_ops(N_OPS, SEED)
    seen = 0
    for i, op in enumerate(ops):
        a, b = apply(py, op), apply(rs, op)
        assert a == b and type(a) is type(b), f"op {i} {op}: returned {a!r} vs {b!r}"

        pe, re_ = py.events_since(seen), rs.events_since(seen)
        assert len(pe) == len(re_), f"op {i} {op}: {len(pe)} vs {len(re_)} new events"
        for x, y in zip(pe, re_):
            assert type(y) is tuple, f"op {i}: rust event is {type(y).__name__}, not tuple"
            assert x[:-1] == y[:-1], f"op {i} {op}:\n  python {x}\n  rust   {y}"
        seen += len(pe)

        if i % CHECK_EVERY == 0 or i == len(ops) - 1:
            assert state(py) == state(rs), f"state differs after op {i}"
            for oid in range(1, seen + 2, 37):
                assert py.has_order(oid) == rs.has_order(oid), f"has_order({oid})"

    # Both must refuse the same bad input, with the same exception type.
    for bad in (("L", "X", 100, 1), ("L", "B", 100, 0), ("L", "B", 100, -3),
                ("L", "B", 100, 0.5), ("L", "B", 100.5, 1), ("M", "B", 0),
                ("L", "B", 100.0, 1), ("L", "B", 100, 1.0),      # whole floats
                ("L", "B", 0, 1), ("L", "S", -5, 1),             # non-positive prices
                ("M", "S", 2.5), ("M", "", 1)):
        for book in (py, rs):
            try:
                apply(book, bad)
                raise AssertionError(f"{type(book).__module__} accepted {bad}")
            except ValueError:
                pass
    assert state(py) == state(rs), "rejected orders left residue"
    return seen


def demo():
    try:
        from bullpen_engine import Book as RustBook
    except ImportError:
        # A skipped test reports success, which is how a typo once hid 154
        # tests in this repo. CI builds the extension and sets this variable,
        # so there a missing module is a failure, not a skip.
        if os.environ.get("BULLPEN_REQUIRE_RUST") == "1":
            raise
        print("skip test_pyo3: bullpen_engine not built "
              "(cd engine/rust; maturin develop --release)")
        return
    n = run(RustBook)
    print(f"ok — {N_OPS} ops, {n} events identical between Python and Rust (PyO3)")


if __name__ == "__main__":
    demo()
    sys.exit(0)
