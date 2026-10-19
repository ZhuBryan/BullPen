"""
The fastest order book available, behind one name.

    from engine.fastbook import Book, ENGINE

`Book` is the Rust engine (`bullpen_engine`, built with maturin, see
engine/rust/README.md) when that extension is installed, otherwise the pure
Python engine in orderbook.py. Both implement the same portable interface
(SPEC.md section 7.1), and test_pyo3.py checks they produce identical event
logs, so callers never need to know which one they got. `ENGINE` says which,
for logs and the /api/metrics gauge.

Set BULLPEN_ENGINE=python to force the Python engine, e.g. to rule the Rust
build out when debugging.

The two constructors differ, and only the constructor: Python takes
`Book(clock=...)`, Rust takes `Book(deterministic=False)`. `Book()` works on
both.
"""

import os
import warnings

try:
    from engine.orderbook import Book as PyBook
except ImportError:
    from orderbook import Book as PyBook

# SPEC.md 7.1: the only methods simulator / backtester code may call.
# submit_many(ops) batches limit/market/cancel calls into one (see orderbook.py).
PORTABLE = ("limit", "market", "cancel", "best_bid", "best_ask", "depth",
            "event_count", "events_since", "has_order", "order_ids", "submit_many")

Book, ENGINE = PyBook, "python"
if os.environ.get("BULLPEN_ENGINE", "").lower() != "python":
    try:
        from bullpen_engine import Book as _RustBook
    except ImportError:
        pass
    else:
        _missing = [m for m in PORTABLE if not callable(getattr(_RustBook, m, None))]
        if _missing:
            # An extension built from older sources: using it would crash the
            # first caller of the missing method, so run on Python instead,
            # loudly. Rebuild with `python -m pip install .` in engine/rust.
            warnings.warn(f"bullpen_engine is out of date (lacks {_missing}); "
                          f"using the Python engine. Rebuild it in engine/rust.")
        else:
            Book, ENGINE = _RustBook, "rust"


def demo():
    missing = [m for m in PORTABLE if not callable(getattr(Book, m, None))]
    assert not missing, f"{ENGINE} Book lacks {missing}"

    b = Book()
    ask = b.limit("S", 100, 10)
    bid = b.limit("B", 100, 4)          # crosses: trades 4 at the resting price
    assert (ask, bid) == (1, 2), "ids start at 1 and count up"
    assert b.best_ask() == 100 and b.best_bid() is None
    assert b.depth("S") == [(100, 6)] and b.depth("B") == []
    assert b.has_order(ask) and not b.has_order(bid)
    assert b.order_ids() == [ask]

    n = b.event_count()
    assert b.cancel(999) is False       # unknown id: logged, not raised
    assert b.cancel(ask) is True
    new = b.events_since(n)
    assert [e[0] for e in new] == ["reject_cancel", "canceled"]
    assert new[0][:2] == ("reject_cancel", 999) and new[1][:3] == ("canceled", ask, 6)
    assert [e[-2] for e in b.events_since(0)] == list(range(1, b.event_count() + 1))

    trades = [e for e in b.events_since(0) if e[0] == "trade"]
    assert [t[:5] for t in trades] == [("trade", 100, 4, bid, ask)]
    assert b.market("B", 5) == 3 and b.best_bid() is None, "market orders never rest"

    # submit_many: one call, same results as the single calls in sequence.
    n = b.event_count()
    got = b.submit_many([("L", "S", 101, 5), ("L", "S", 102, 5), ("M", "B", 7),
                         ("X", 5), ("X", 5), ("L", "B", 99, 1)])
    assert got == [4, 5, 6, True, False, 7], got
    assert [type(x) for x in got[3:5]] == [bool, bool]
    assert [e[0] for e in b.events_since(n)] == [
        "accepted", "accepted", "accepted", "trade", "trade", "canceled",
        "reject_cancel", "accepted"]
    assert b.depth("S") == [] and b.depth("B") == [(99, 1)] and b.submit_many([]) == []
    # A bad op raises like its single call; the ops before it stay applied.
    for bad_ops in ([("L", "B", 98, 1), ("L", "B", 100, 0)], [("L", "B", 98, 1), ("Q", 1)],
                    [("L", "B", 98, 1), ("M", "B")], [("L", "B", 98, 1), ["X", 1]]):
        before = b.event_count()
        try:
            b.submit_many(bad_ops)
            raise AssertionError(f"bad batch accepted: {bad_ops}")
        except ValueError:
            pass
        assert [e[0] for e in b.events_since(before)] == ["accepted"], "first op must stand"

    for bad in (lambda: b.limit("X", 100, 1), lambda: b.limit("B", 100, 0),
                lambda: b.limit("B", 100, 0.5), lambda: b.limit("B", 100.5, 1),
                lambda: b.market("B", -1)):
        try:
            bad()
            raise AssertionError("bad order accepted")
        except ValueError:
            pass
    print(f"ok — fastbook using the {ENGINE} engine")


if __name__ == "__main__":
    demo()
