"""
Differential test: prove the Rust engine behaves exactly like the Python one.

A rewrite that is fast but subtly different is worse than no rewrite, because
every result built on it is quietly wrong. So the Python engine is treated as
the oracle and the Rust port has to reproduce its event log exactly.

Both engines are fed the SAME op file rather than each generating its own
stream, which sidesteps the fact that two languages' RNGs will never agree.
Order ids are assigned sequentially by each engine, so identical input yields
identical ids and the logs are directly comparable.

Usage:
    python diff_test.py gen ops.txt 20000     # generate a random op stream
    python diff_test.py replay ops.txt        # print Python's event log
    python diff_test.py check ops.txt         # generate, run both, diff

`check` requires the Rust binary to be built:
    cd rust && cargo build --release
"""

import random
import subprocess
import sys
from pathlib import Path

try:
    from engine.orderbook import BUY, SELL, Book
except ImportError:
    from orderbook import BUY, SELL, Book

RUST_BIN = Path(__file__).parent / "rust" / "target" / "release" / "bullpen_engine"


def generate(n, seed=42):
    """Random but valid op stream. Cancels only reference ids already issued."""
    rnd = random.Random(seed)
    ops, live, next_id = [], [], 1
    for _ in range(n):
        side = rnd.choice(["B", "S"])
        roll = rnd.randint(0, 99)
        if roll < 15 and live:
            ops.append(f"X {live.pop(rnd.randrange(len(live)))}")
        elif roll < 25:
            ops.append(f"M {side} {rnd.randint(1, 100)}")
            next_id += 1
        else:
            off = rnd.randint(1, 12)
            price = 10_000 - off if side == "B" else 10_000 + off
            ops.append(f"L {side} {price} {rnd.randint(1, 100)}")
            live.append(next_id)
            next_id += 1
    return ops


def encode(ev):
    """Must match Event::encode in rust/src/lib.rs exactly."""
    kind = ev[0]
    if kind == "accepted":
        _, oid, side, price, qty = ev
        return f"A {oid} {side} {'M' if price is None else price} {qty}"
    if kind == "trade":
        _, price, qty, aggressor, resting = ev
        return f"T {price} {qty} {aggressor} {resting}"
    if kind == "canceled":
        _, oid, remaining = ev
        return f"C {oid} {remaining}"
    if kind == "reject_cancel":
        return f"R {ev[1]}"
    raise ValueError(f"unknown event {ev!r}")


def replay(path):
    book = Book()
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        if f[0] == "L":
            book.limit(BUY if f[1] == "B" else SELL, int(f[2]), int(f[3]))
        elif f[0] == "M":
            book.market(BUY if f[1] == "B" else SELL, int(f[2]))
        elif f[0] == "X":
            book.cancel(int(f[1]))
        else:
            raise ValueError(f"unknown op {f[0]}")
    return [encode(e) for e in book.events]


def check(path, n=20_000, seed=42):
    Path(path).write_text("\n".join(generate(n, seed)) + "\n")
    print(f"generated {n} ops -> {path}")

    py = replay(path)
    print(f"python: {len(py)} events")

    if not RUST_BIN.exists() and not RUST_BIN.with_suffix(".exe").exists():
        print(f"\nRust binary not built. Run:\n  cd rust && cargo build --release")
        return 1

    binary = RUST_BIN if RUST_BIN.exists() else RUST_BIN.with_suffix(".exe")
    out = subprocess.run([str(binary), "replay", path],
                         capture_output=True, text=True, check=True)
    rs = out.stdout.splitlines()
    print(f"rust:   {len(rs)} events")

    if py == rs:
        print(f"\nIDENTICAL — {len(py)} events matched exactly.")
        return 0

    print("\nMISMATCH")
    if len(py) != len(rs):
        print(f"  event count differs: python {len(py)} vs rust {len(rs)}")
    for i, (a, b) in enumerate(zip(py, rs)):
        if a != b:
            print(f"  first difference at event {i}:")
            print(f"    python: {a}")
            print(f"    rust:   {b}")
            lo = max(0, i - 3)
            print("  context (python):")
            for j in range(lo, min(len(py), i + 4)):
                print(f"    {j:>6} {py[j]}")
            break
    return 1


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    path = sys.argv[2] if len(sys.argv) > 2 else "ops.txt"
    if mode == "gen":
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 20_000
        Path(path).write_text("\n".join(generate(n)) + "\n")
        print(f"wrote {n} ops to {path}")
    elif mode == "replay":
        print("\n".join(replay(path)))
    else:
        sys.exit(check(path, int(sys.argv[3]) if len(sys.argv) > 3 else 20_000))
