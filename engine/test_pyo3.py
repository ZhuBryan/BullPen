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

A second pass (run_batched) drives another 20,000-op stream through
submit_many in random-sized chunks, on both engines, against a Python book
fed the same ops one call at a time: results, result types, events and state
must all match, and a bad op inside a batch must raise the same ValueError,
with the ops before it applied and the ops after it not.

`python -m engine.test_pyo3 --standin` runs both passes against a Python
stand-in for the Rust class, to check the harness itself without a build.

Skips cleanly (exit 0) when the extension is not built:
    cd rust; python -m pip install .
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


# ---- submit_many ----------------------------------------------------------
# The batch call must be indistinguishable from the single calls in sequence,
# on each engine and across them. Three books get the same stream: a Python
# book driven one call at a time (the reference), and a Python and a Rust
# book driven through submit_many in random-sized chunks.

def run_batched(rust_book_cls, seed=SEED + 1):
    ref = PyBook(clock=counter_clock())
    pyb, rs = PyBook(clock=counter_clock()), rust_book_cls(deterministic=True)
    ops = gen_ops(N_OPS, seed)
    rnd = random.Random(seed)
    i = seen = chunks = 0
    while i < len(ops):
        chunk = ops[i:i + rnd.choice((1, 2, 3, 5, 8, 13, 40, 200))]
        want = [apply(ref, op) for op in chunk]
        got_py, got_rs = pyb.submit_many(chunk), rs.submit_many(chunk)
        assert type(got_py) is list and type(got_rs) is list
        for k, (w, a, b) in enumerate(zip(want, got_py, got_rs)):
            assert w == a == b and type(w) is type(a) is type(b), (
                f"op {i + k} {chunk[k]}: single {w!r}, python batch {a!r}, rust batch {b!r}")
        assert len(want) == len(got_py) == len(got_rs), f"chunk at op {i}: result count"

        # Python batch vs Python singles: identical including ts (both use a
        # counter clock). Rust vs Python: identical except ts, as in run().
        er, ep, ers = ref.events_since(seen), pyb.events_since(seen), rs.events_since(seen)
        assert er == ep, f"chunk at op {i}: python batch events differ from single calls"
        assert len(er) == len(ers), f"chunk at op {i}: {len(er)} vs {len(ers)} new events"
        for x, y in zip(er, ers):
            assert x[:-1] == y[:-1], f"chunk at op {i}:\n  python {x}\n  rust   {y}"
        seen += len(er)
        i += len(chunk)
        chunks += 1
        if chunks % 50 == 0 or i == len(ops):
            assert state(ref) == state(pyb) == state(rs), f"state differs after op {i}"

    # Bad input inside a batch: the same ValueError as the single call, the
    # ops before it applied, the ops after it not. This mirrors the single
    # calls exactly: they would have applied the earlier ops and then raised.
    good_before, good_after = ("L", "B", 500, 3), ("L", "S", 1_500, 4)
    bad_ops = [("L", "X", 100, 1), ("L", "B", 100, 0), ("L", "B", 100, -3),
               ("L", "B", 100, 0.5), ("L", "B", 100.5, 1), ("M", "B", 0),
               ("L", "B", 100.0, 1), ("L", "B", 100, 1.0), ("L", "B", 0, 1),
               ("L", "S", -5, 1), ("M", "S", 2.5), ("M", "", 1)]
    # Malformed commands: no single-call equivalent, so both engines must
    # simply agree (ValueError, earlier ops applied).
    malformed = [("Q", 1), ("L", "B", 100), ("M", "B"), ("X",), (), ("M", "B", 1, 2),
                 ("X", 1, 2), (5, "B", 1), ("l", "B", 100, 1), ["L", "B", 100, 1], "L", None]
    for bad, has_single in [(b, True) for b in bad_ops] + [(b, False) for b in malformed]:
        n0 = ref.event_count()
        apply(ref, good_before)
        if has_single:
            try:
                apply(ref, bad)
                raise AssertionError(f"single call accepted {bad}")
            except ValueError:
                pass
        for book in (pyb, rs):
            try:
                book.submit_many([good_before, bad, good_after])
                raise AssertionError(f"{type(book).__module__} batch accepted {bad!r}")
            except ValueError:
                pass
        new = ref.events_since(n0)
        # good_before was applied (it may also trade), good_after was not.
        assert [e[0] for e in new].count("accepted") == 1, new
        assert pyb.events_since(n0) == new, f"python batch after {bad!r}"
        assert [e[:-1] for e in rs.events_since(n0)] == [e[:-1] for e in new], \
            f"rust batch after {bad!r}"
        assert state(ref) == state(pyb) == state(rs), f"state differs after bad batch {bad!r}"
    assert pyb.submit_many([]) == [] and rs.submit_many([]) == []
    # Any iterable of ops, as with a list.
    gen = (op for op in [("L", "B", 600, 1), ("X", 10**9)])
    assert pyb.submit_many(gen) == rs.submit_many(iter([("L", "B", 600, 1), ("X", 10**9)]))
    assert state(pyb) == state(rs)
    return chunks, seen


def _standin():
    """A Python class shaped like the Rust one: Book(deterministic=True).

    Running the harness against it checks the harness itself (and that
    submit_many on the Python engine equals single calls) without a Rust
    build: `python -m engine.test_pyo3 --standin`.
    """
    class StandIn(PyBook):
        def __init__(self, deterministic=False):
            super().__init__(clock=counter_clock() if deterministic else None)
    StandIn.__module__ = "standin"
    return StandIn


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
              "(cd engine/rust; python -m pip install .)")
        return
    n = run(RustBook)
    print(f"ok — {N_OPS} ops, {n} events identical between Python and Rust (PyO3)")
    chunks, n = run_batched(RustBook)
    print(f"ok — submit_many: {N_OPS} ops in {chunks} batches, {n} events identical "
          f"to single calls on both engines; bad ops raise identically")


if __name__ == "__main__":
    if "--standin" in sys.argv:
        StandIn = _standin()
        n = run(StandIn)
        chunks, m = run_batched(StandIn)
        print(f"ok — harness vs Python stand-in: {n} events (single calls), "
              f"{m} events in {chunks} batches")
    else:
        demo()
    sys.exit(0)
