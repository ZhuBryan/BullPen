"""
fuzz.py: throw hostile input at the engines and parsers until something breaks.

diff_test.py checks the three engines agree on ONE friendly order stream: a
few prices around 10,000, sizes 1-100, cancels of orders that exist. Real
bugs live where nobody looked: a price of 1, a price of 2^63 - 1, a cancel of
an order that already filled, a market order into an empty book, one price
level with a thousand orders queued on it. This file generates streams like
that, thousands of them, and checks two things on each:

  1. Differential: the Python engine (the oracle) and every compiled engine
     (C++ and Rust, if built) print byte-identical event logs. The Python
     book's invariants are checked after every single op as well.
  2. Parsers: random and mutated bytes fed to the FIX decoder, the FIX frame
     reader, the write-ahead log, the capture-file reader and the C++ op-file
     parser. A parser may REJECT garbage, with the error type its contract
     names; it may never crash, hang or raise anything else.

When a stream makes the engines disagree, it is SHRUNK before being saved:
chunks and then single ops are deleted for as long as the failure still
happens. A 2,000-op failure nobody can read becomes a 4-op one anybody can.
Any subsequence of a valid stream is itself valid (a cancel of a missing id
is a legal op), which is what makes deleting ops safe.

    python fuzz.py                     # 60 s campaign, engines + parsers
    python fuzz.py --seconds 600 --seed 7
    python fuzz.py --engines-only      # or --parsers-only
    python fuzz.py --demo              # self-check (also run by test.py)

The C++ engine is built here from cpp/ (optimised + AddressSanitizer builds)
in a temporary folder when g++ and make exist, so this runs on its own.
"""

import argparse
import asyncio
import gzip
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path

try:
    from engine import fix, fix_session, replay as capture_replay, wal
    from engine.diff_test import CPP_BIN, RUST_BIN, encode, find_binary
    from engine.orderbook import Book, _check_invariants, counter_clock
except ImportError:
    import fix, fix_session, wal                      # noqa: E401
    import replay as capture_replay
    from diff_test import CPP_BIN, RUST_BIN, encode, find_binary
    from orderbook import Book, _check_invariants, counter_clock

HERE = Path(__file__).parent
INT64_MAX = 2**63 - 1          # the largest price or qty the compiled engines can hold
UINT64_MAX = 2**64 - 1         # the largest order id

# ---------------------------------------------------------------------------
# 1. Hostile order streams
# ---------------------------------------------------------------------------
EXTREME_PRICES = [1, 2, 2**53, 2**53 + 1, INT64_MAX - 1, INT64_MAX]
EXTREME_QTYS = [1, 2, 10**9, 2**62, INT64_MAX]


def gen_stream(rnd, n):
    """n op-file lines, mixing friendly traffic with the cases that break engines."""
    ops, live, gone, next_id = [], [], [], 1
    hot = rnd.choice([1, 10_000, INT64_MAX - 5])          # the hammered level
    # Each stream gets its own mix, so some are 90% cancels and some all sweeps.
    w = [rnd.random() for _ in range(7)]
    while len(ops) < n:
        kind = rnd.choices(range(7), weights=w)[0]
        side = rnd.choice("BS")
        qty = rnd.choice(EXTREME_QTYS) if rnd.random() < 0.1 else rnd.randint(1, 100)
        if kind == 0:                                   # same price, over and over
            p = hot + rnd.randint(-1, 1) if hot < INT64_MAX - 5 else hot + rnd.randint(0, 5)
            ops.append(f"L {side} {max(p, 1)} {qty}")
        elif kind == 1:                                 # extreme prices
            ops.append(f"L {side} {rnd.choice(EXTREME_PRICES)} {qty}")
        elif kind == 2:                                 # market order, maybe into nothing
            ops.append(f"M {side} {qty}")
        elif kind == 3:                                 # big order sweeping many levels
            far = 1 if side == "S" else INT64_MAX
            ops.append(f"L {side} {far} {rnd.choice([10**6, INT64_MAX])}")
        elif kind == 4 and live:                        # burst of cancels of live orders
            for _ in range(rnd.randint(1, 20)):
                if live:
                    oid = live.pop(rnd.randrange(len(live)))
                    gone.append(oid)
                    ops.append(f"X {oid}")
            continue
        elif kind == 5:                                 # cancels that must be rejected
            bad = [0, UINT64_MAX, next_id, next_id + rnd.randint(1, 1000)]
            if gone:
                bad.append(rnd.choice(gone))            # already filled or cancelled
            ops.append(f"X {rnd.choice(bad)}")
            continue
        else:                                           # ordinary traffic
            off = rnd.randint(1, 12)
            ops.append(f"L {side} {10_000 - off if side == 'B' else 10_000 + off} {qty}")
        if ops[-1][0] == "L":
            live.append(next_id)
        next_id += 1
    return ops[:n]


# ---------------------------------------------------------------------------
# Engines: each takes a list of op lines and returns the encoded event log,
# or raises. The Python oracle also checks its invariants after every op.
# ---------------------------------------------------------------------------
def python_engine(ops, book_cls=Book):
    book = book_cls(clock=counter_clock())
    for op in ops:
        wal.apply_op(book, op)
        _check_invariants(book)
    return [encode(e) for e in book.events]


def compiled_engine(binary, workdir):
    def run(ops):
        path = os.path.join(workdir, "fuzz_ops.txt")
        Path(path).write_text("\n".join(ops) + "\n")
        out = subprocess.run([str(binary), "replay", path], capture_output=True,
                             text=True, timeout=60)
        if out.returncode != 0:
            raise RuntimeError(f"exit {out.returncode}: {out.stderr.strip()[:300]}")
        return out.stdout.splitlines()
    return run


def disagreement(ops, engines):
    """None if every engine agrees with the oracle on `ops`, else a description."""
    try:
        want = python_engine(ops)
    except Exception as e:                      # oracle crashed or broke an invariant
        return f"python: {type(e).__name__}: {e}"
    for name, run in engines:
        try:
            got = run(ops)
        except Exception as e:
            return f"{name}: {type(e).__name__}: {e}"
        if got != want:
            i = next((i for i, (a, b) in enumerate(zip(want, got)) if a != b),
                     min(len(want), len(got)))
            a = want[i] if i < len(want) else "<missing>"
            b = got[i] if i < len(got) else "<missing>"
            return f"{name}: event {i} differs: python {a!r}, {name} {b!r}"
    return None


def shrink(ops, fails):
    """Delete chunks, then single ops, while `fails(ops)` stays true (delta debugging)."""
    chunk = max(1, len(ops) // 2)
    while True:
        changed, i = False, 0
        while i < len(ops):
            cand = ops[:i] + ops[i + chunk:]
            if cand and fails(cand):
                ops, changed = cand, True       # keep the smaller one, retry same spot
            else:
                i += chunk
        if not changed:
            if chunk == 1:
                return ops
            chunk = max(1, chunk // 2)


def fuzz_engines(engines, seconds, seed, max_ops=400, out_dir=".", quiet=False):
    """Run streams until time is up. Returns (stats, findings: list of (why, ops, path))."""
    rnd = random.Random(seed)
    stats = {"streams": 0, "ops": 0}
    findings, stop = [], time.time() + seconds
    while time.time() < stop:
        ops = gen_stream(rnd, rnd.randint(1, max_ops))
        stats["streams"] += 1
        stats["ops"] += len(ops)
        why = disagreement(ops, engines)
        if why is None:
            continue
        small = shrink(ops, lambda o: disagreement(o, engines) is not None)
        path = os.path.join(out_dir, f"fuzz_fail_{seed}_{stats['streams']}.txt")
        Path(path).write_text(f"# {disagreement(small, engines)}\n" + "\n".join(small) + "\n")
        findings.append((why, small, path))
        if not quiet:
            print(f"FOUND after {stats['streams']} streams: {why}\n"
                  f"  shrunk {len(ops)} -> {len(small)} ops, saved to {path}")
        if len(findings) >= 5:
            break
    return stats, findings


# ---------------------------------------------------------------------------
# 2. Parsers. Each target takes bytes and either returns or raises one of the
# exception types it is allowed to raise. Anything else is a finding.
# ---------------------------------------------------------------------------
def _fix_seeds():
    return [fix.new_order_single("C1", "AAPL", fix.Side.BUY, 100, price=150.25),
            fix.cancel_request("C2", "C1", "AAPL", fix.Side.BUY, 100),
            fix.reject(3, "bad", text="no")]


def mutate(rnd, data, alphabet=b"=\x01|0123456789AZaz \n"):
    """A few random edits: flip, insert, delete, duplicate a slice, truncate."""
    b = bytearray(data)
    for _ in range(rnd.randint(1, 6)):
        r = rnd.random()
        i = rnd.randrange(len(b) + 1)
        if r < 0.25 and b:
            b[min(i, len(b) - 1)] = rnd.randrange(256)
        elif r < 0.5:
            b[i:i] = bytes([rnd.choice(alphabet)])
        elif r < 0.7 and b:
            del b[i:i + rnd.randint(1, 4)]
        elif r < 0.85 and b:
            j = rnd.randrange(len(b))
            b[i:i] = b[j:j + rnd.randint(1, 16)]
        else:
            del b[i:]
    return bytes(b)


def target_fix_decode(data):
    fix.decode(data.decode("latin-1"))          # contract: FixError on bad input


def target_read_frame(data):
    async def go():
        r = asyncio.StreamReader()
        r.feed_data(data)
        r.feed_eof()
        while True:                             # read frames until the stream runs dry
            raw, _ = await fix_session.read_frame(r)
            try:
                fix.decode(raw)
            except fix.FixError:
                pass
    asyncio.run(go())


def target_wal(data, workdir):
    path = os.path.join(workdir, "fuzz.wal")
    Path(path).write_bytes(data)
    book, n, good, tail = wal.recover(path)      # contract: never raises
    assert tail in ("clean", "torn", "corrupt") and 0 <= good <= len(data)


def target_capture(data, workdir):
    path = os.path.join(workdir, "fuzz.jsonl.gz")
    Path(path).write_bytes(data)
    real, sys.stderr = sys.stderr, open(os.devnull, "w")   # it prints a note per cut-off file
    try:
        for _ in capture_replay._records(path):  # contract: stops at damage, never raises
            pass
    finally:
        sys.stderr.close()
        sys.stderr = real


def target_cpp_parser(binary, workdir):
    def run(data):
        path = os.path.join(workdir, "fuzz_parse.txt")
        Path(path).write_bytes(data)
        out = subprocess.run([str(binary), "replay", path], capture_output=True, timeout=60)
        err = out.stderr.decode("latin-1")
        # 0 = replayed, 1 = rejected with a message. A signal (negative code)
        # or a sanitizer report is a crash, whatever the exit code.
        if out.returncode not in (0, 1) or "Sanitizer" in err or "runtime error" in err:
            raise AssertionError(f"exit {out.returncode}: {err.strip()[:400]}")
        if out.returncode == 1 and not err.strip():
            raise AssertionError("rejected without saying why")
    return run


def _wal_seed(rnd):
    """A valid log, then (sometimes) records whose payload is junk but whose
    CRC is correct, so the fuzz reaches the code that trusts a checked record."""
    lines = [wal._record(i + 1, op) for i, op in enumerate(gen_stream(rnd, rnd.randint(1, 30)))]
    if rnd.random() < 0.5:
        junk = mutate(rnd, rnd.choice(lines).split(b" ", 2)[2].rstrip(b"\n"))
        payload = f"{len(lines) + 1} ".encode() + junk.replace(b"\n", b" ")
        lines.append(b"%08x %s\n" % (zlib.crc32(payload), payload))
    return b"".join(lines)


def _capture_seed(rnd):
    rows = [b'{"e":"depthUpdate","U":%d,"u":%d,"b":[["100.0","1"]],"a":[]}\n' % (i, i)
            for i in range(rnd.randint(0, 20))]
    raw = b"".join(rows)
    if rnd.random() < 0.5:
        raw = mutate(rnd, raw) if raw else raw            # damage inside the text
    blob = gzip.compress(raw)
    return mutate(rnd, blob) if rnd.random() < 0.7 else blob   # or the gzip itself


def parser_targets(workdir, cpp_asan=None):
    """(name, function(bytes), allowed exception types, seed generator)."""
    rnd_seed = lambda rnd: rnd.choice(_fix_seeds()).encode("latin-1")
    t = [
        ("fix.decode", target_fix_decode, (fix.FixError,), rnd_seed),
        ("fix_session.read_frame", target_read_frame,
         (fix_session.Garbled, asyncio.IncompleteReadError, asyncio.LimitOverrunError),
         lambda rnd: b"".join(rnd_seed(rnd) for _ in range(rnd.randint(1, 3)))),
        ("wal.recover", lambda d: target_wal(d, workdir), (), _wal_seed),
        ("replay._records", lambda d: target_capture(d, workdir), (), _capture_seed),
    ]
    if cpp_asan:
        t.append(("c++ op parser (ASan)", target_cpp_parser(cpp_asan, workdir), (),
                  lambda rnd: "\n".join(gen_stream(rnd, rnd.randint(1, 20))).encode()))
    return t


def fuzz_parsers(targets, seconds, seed, quiet=False):
    """Round-robin over targets until time is up. Returns (stats, findings)."""
    rnd = random.Random(seed)
    stats = {name: 0 for name, *_ in targets}
    findings, seen, stop = [], set(), time.time() + seconds
    while time.time() < stop and len(findings) < 20:
        for name, fn, allowed, seed_gen in targets:
            r = rnd.random()
            if r < 0.15:                                  # pure noise
                data = bytes(rnd.randrange(256) for _ in range(rnd.randint(0, 200)))
            else:                                         # a valid input, damaged
                data = seed_gen(rnd)
                if r < 0.9 and name not in ("wal.recover", "replay._records"):
                    data = mutate(rnd, data)
            stats[name] += 1
            try:
                fn(data)
            except allowed:
                pass
            except Exception as e:
                key = (name, type(e).__name__, str(e)[:60])
                if key not in seen:                       # report each distinct failure once
                    seen.add(key)
                    findings.append((name, f"{type(e).__name__}: {e}"[:400], data))
                    if not quiet:
                        print(f"FOUND in {name}: {type(e).__name__}: {str(e)[:200]}\n"
                              f"  input: {data[:120]!r}")
    return stats, findings


# ---------------------------------------------------------------------------
# Building the C++ engine for the campaign
# ---------------------------------------------------------------------------
def build_cpp(workdir):
    """(release binary, ASan binary) built from cpp/ into workdir, or (None, None)."""
    if not (shutil.which("make") and shutil.which("g++")):
        return None, None
    dst = os.path.join(workdir, "cpp")
    shutil.copytree(HERE / "cpp", dst, ignore=shutil.ignore_patterns("bullpen_cpp*"))
    for target in ("all", "debug"):
        r = subprocess.run(["make", "-s", target], cwd=dst, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"c++ build failed ({target}):\n{r.stderr[:800]}", file=sys.stderr)
            return None, None
    return Path(dst) / "bullpen_cpp", Path(dst) / "bullpen_cpp_debug"


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
def demo():
    t0 = time.time()

    # The fuzzer must find a planted bug. This engine breaks FIFO, but only
    # when a third order joins a level: it jumps to the front of the queue.
    # The friendly stream in diff_test.py would catch this too; the point is
    # that the fuzzer finds it AND shrinks the evidence to a handful of ops.
    class QueueJumpBook(Book):
        def _rest(self, o):
            super()._rest(o)
            level = self._side[o.side][0][o.price]
            if len(level) >= 3:
                level.rotate(1)                  # newest order to the front

    buggy = [("queue-jump", lambda ops: python_engine(ops, QueueJumpBook))]
    with tempfile.TemporaryDirectory() as d:
        stats, found = fuzz_engines(buggy, seconds=5, seed=1, max_ops=200, out_dir=d, quiet=True)
        assert found, "fuzzer missed a planted FIFO bug"
        why, small, path = found[0]
        assert len(small) <= 6, f"shrunk only to {len(small)} ops: {small}"
        assert disagreement(small, buggy) is not None and os.path.exists(path)
        limits = [op for op in small if op.startswith("L")]
        assert len(limits) >= 3, small          # needs three resting orders to show
        n_small = len(small)

        # The real engines agree with themselves on hostile streams (Python
        # vs Python catches nondeterminism and invariant breaks).
        same = [("python again", python_engine)]
        stats, found = fuzz_engines(same, seconds=1.5, seed=2, out_dir=d, quiet=True)
        assert not found and stats["streams"] > 5, (stats, found)

        # Parsers: a short pass must find nothing, and a planted parser bug
        # (a crash on one specific byte) must be found.
        stats, found = fuzz_parsers(parser_targets(d), seconds=1.5, seed=3, quiet=True)
        assert not found, found

        def fragile(data):
            if b"=" in data[:3]:
                raise KeyError("planted")
        stats, found = fuzz_parsers([("fragile", fragile, (), lambda r: b"8=FIX")],
                                    seconds=0.3, seed=4, quiet=True)
        assert found and found[0][0] == "fragile", found

    # The shrinker on its own: the minimal failing subsequence is exactly found.
    small = shrink(list(range(100)), lambda o: 17 in o and 42 in o)
    assert small == [17, 42], small

    print(f"ok — fuzz: planted FIFO bug found and shrunk to {n_small} ops, planted "
          f"parser crash found, hostile streams clean ({time.time() - t0:.1f}s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--seed", type=int, default=int(time.time()) % 100_000)
    ap.add_argument("--max-ops", type=int, default=400)
    ap.add_argument("--engines-only", action="store_true")
    ap.add_argument("--parsers-only", action="store_true")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        return demo()

    print(f"seed {a.seed} (rerun with --seed {a.seed} to reproduce)")
    with tempfile.TemporaryDirectory() as d:
        cpp, cpp_asan = build_cpp(d)
        engines = []
        for name, binary in (("c++", cpp or find_binary(CPP_BIN)), ("rust", find_binary(RUST_BIN))):
            if binary is None:
                continue
            try:                                # a binary built for another OS (a
                compiled_engine(binary, d)(["M B 1"])   # Windows .exe seen from Linux)
            except OSError as e:
                print(f"{name}: {binary.name} cannot run here ({e.strerror}), skipped")
                continue
            engines.append((name, compiled_engine(binary, d)))
        print("engines: python (oracle)" + "".join(f", {n}" for n, _ in engines))
        bad = 0
        share = 0.5 if not (a.engines_only or a.parsers_only) else 1.0
        if not a.parsers_only:
            stats, found = fuzz_engines(engines, a.seconds * share, a.seed, a.max_ops, out_dir=".")
            print(f"engines: {stats['streams']:,} streams, {stats['ops']:,} ops, "
                  f"{len(found)} finding(s)")
            bad += len(found)
        if not a.engines_only:
            stats, found = fuzz_parsers(parser_targets(d, cpp_asan), a.seconds * share, a.seed)
            print("parsers: " + ", ".join(f"{k} {v:,}" for k, v in stats.items())
                  + f"; {len(found)} finding(s)")
            bad += len(found)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
