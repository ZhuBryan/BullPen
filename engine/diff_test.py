"""
Differential test: prove the Rust and C++ engines behave exactly like the
Python one.

A rewrite that is fast but subtly different is worse than no rewrite, because
every result built on it is quietly wrong. So the Python engine is treated as
the oracle and each compiled port has to reproduce its event log exactly.

Both engines are fed the SAME op file rather than each generating its own
stream, which sidesteps the fact that two languages' RNGs will never agree.
Order ids are assigned sequentially by each engine, so identical input yields
identical ids and the logs are directly comparable.

Usage:
    python diff_test.py gen ops.txt 20000     # generate a random op stream
    python diff_test.py replay ops.txt        # print Python's event log
    python diff_test.py check ops.txt         # generate, run both, diff

`check` compares Python against every compiled engine that is built, and
reports each separately. It fails if neither is built, or if any built one
disagrees with Python:
    cd rust && cargo build --release      # Rust
    cd cpp && make                        # C++
"""

import random
import subprocess
import sys
from pathlib import Path

try:
    from engine.orderbook import BUY, SELL, Book, counter_clock
except ImportError:
    from orderbook import BUY, SELL, Book, counter_clock

HERE = Path(__file__).parent
RUST_BIN = HERE / "rust" / "target" / "release" / "bullpen_engine"
CPP_BIN = HERE / "cpp" / "bullpen_cpp"

# (name, binary without suffix, build command shown when it is missing)
ENGINES = [
    ("rust", RUST_BIN, "cd rust && cargo build --release"),
    ("c++", CPP_BIN, "cd cpp && make"),
]


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
    """Must match Event::encode in rust/src/lib.rs exactly.

    `seq` is compared because it is deterministic: the same input must produce
    the same sequence numbers in both engines. The timestamp (ev[-1]) is NOT
    compared, because a clock reading is process-local and the two engines can
    never agree on it. Diffing it would fail for a reason that says nothing
    about whether the matching logic agrees.
    """
    kind, seq = ev[0], ev[-2]
    if kind == "accepted":
        oid, side, price, qty = ev[1], ev[2], ev[3], ev[4]
        return f"{seq} A {oid} {side} {'M' if price is None else price} {qty}"
    if kind == "trade":
        return f"{seq} T {ev[1]} {ev[2]} {ev[3]} {ev[4]}"
    if kind == "canceled":
        return f"{seq} C {ev[1]} {ev[2]}"
    if kind == "reject_cancel":
        return f"{seq} R {ev[1]}"
    raise ValueError(f"unknown event {ev!r}")


def replay(path):
    # Counter clock, matching the Rust side's Book::deterministic(): this log is
    # for comparison, so it must be reproducible rather than clock-dependent.
    book = Book(clock=counter_clock())
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


def find_binary(base):
    """The built binary, with or without a Windows .exe suffix, else None.

    The native name is tried first, so a stale binary from another OS in the
    same tree (say a WSL build seen from Windows) cannot shadow a fresh one.
    """
    exe = base.with_suffix(".exe")
    order = (exe, base) if sys.platform == "win32" else (base, exe)
    for candidate in order:
        if candidate.exists():
            return candidate
    return None


def compare(name, py, other):
    """Report one engine against the Python oracle. True if identical."""
    if py == other:
        print(f"{name}: IDENTICAL — {len(py)} events matched exactly.")
        return True

    print(f"{name}: MISMATCH")
    if len(py) != len(other):
        print(f"  event count differs: python {len(py)} vs {name} {len(other)}")
    for i in range(max(len(py), len(other))):
        a = py[i] if i < len(py) else "<missing>"
        b = other[i] if i < len(other) else "<missing>"
        if a != b:
            print(f"  first difference at event {i}:")
            print(f"    python: {a}")
            print(f"    {name + ':':<7} {b}")
            lo = max(0, i - 3)
            print("  context (python):")
            for j in range(lo, min(len(py), i + 4)):
                print(f"    {j:>6} {py[j]}")
            break
    return False


def check(path, n=20_000, seed=42):
    Path(path).write_text("\n".join(generate(n, seed)) + "\n")
    print(f"generated {n} ops -> {path}")

    py = replay(path)
    print(f"python: {len(py)} events")

    ran, failed = 0, 0
    for name, base, build in ENGINES:
        binary = find_binary(base)
        print()
        if binary is None:
            print(f"{name}: not built, skipped. Run:\n  {build}")
            continue
        try:
            out = subprocess.run([str(binary), "replay", path],
                                 capture_output=True, text=True)
        except OSError as e:
            # Typically a binary built for another OS (a Windows .exe seen
            # from WSL or a Linux container). Nothing was compared, so this
            # counts as not built rather than as a pass or a mismatch.
            print(f"{name}: {binary.name} found but cannot run here ({e.strerror}), skipped.")
            print(f"  rebuild for this platform:\n  {build}")
            continue
        ran += 1
        if out.returncode != 0:
            print(f"{name}: FAILED to replay (exit {out.returncode})")
            print("  " + out.stderr.strip().replace("\n", "\n  "))
            failed += 1
            continue
        other = out.stdout.splitlines()
        print(f"{name + ':':<7} {len(other)} events")
        if not compare(name, py, other):
            failed += 1

    if ran == 0:
        print("\nno compiled engine built, nothing to compare against.")
        return 1
    print(f"\n{ran - failed}/{ran} compiled engine(s) identical to python.")
    return 1 if failed else 0


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
