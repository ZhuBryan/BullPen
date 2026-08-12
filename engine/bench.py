"""
Benchmark harness for the matching engine.

The number that gets quoted in interviews is not "orders per second". It is the
tail: p99 and p99.9. An engine averaging 200ns that stalls for 4ms on a resize
is worse than one that never leaves 800ns, because in trading the slow path is
exactly the path that runs when the market is moving.

Methodology, because an unexplained benchmark is worth nothing:
  * WARMUP first, discarded. Cold caches and lazy allocations are not the
    steady state you are measuring.
  * Two measurements, because they answer different questions:
      - throughput: total wall time / N. No per-op timer overhead, so this is
        the honest ops/sec figure.
      - latency percentiles: each op timed individually. perf_counter_ns has
        roughly 100ns resolution here, so single-digit-ns claims would be
        measuring the clock, not the engine. Timer overhead is measured and
        reported rather than hidden.
  * A FIXED SEED so the order stream is identical across runs. Comparing a
    Python run against a future Rust run on different input is meaningless.

Run `python bench.py` (add `--quick` for a fast pass).
"""

import argparse
import random
import time

from orderbook import BUY, SELL, Book


def make_stream(n, seed=42, mid=10_000, spread=2, cancel_rate=0.15, market_rate=0.10):
    """A synthetic but plausible order stream: mostly limits near the touch,
    some marketable flow, some cancels. Deterministic for a given seed."""
    rnd = random.Random(seed)
    ops = []
    for _ in range(n):
        r = rnd.random()
        side = rnd.choice([BUY, SELL])
        qty = rnd.randint(1, 100)
        if r < cancel_rate:
            ops.append(("cancel", None, None, None))
        elif r < cancel_rate + market_rate:
            ops.append(("market", side, None, qty))
        else:
            # Cluster near the touch, with a fat-ish tail further out.
            off = int(abs(rnd.gauss(0, spread * 2))) + 1
            price = mid - off if side == BUY else mid + off
            ops.append(("limit", side, price, qty))
    return ops


def load_ops(path):
    """Read the shared op-file format used by diff_test.py and the Rust binary.

    Benchmarking two engines on separately generated streams makes the
    workloads merely similar. Feeding both the same file is what makes a
    quoted ratio defensible.
    """
    ops = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        if f[0] == "L":
            ops.append(("limit", BUY if f[1] == "B" else SELL, int(f[2]), int(f[3])))
        elif f[0] == "M":
            ops.append(("market", BUY if f[1] == "B" else SELL, None, int(f[2])))
        elif f[0] == "X":
            ops.append(("cancel_id", None, None, int(f[1])))
    return ops


def _run(book, ops, live):
    """Execute one op; returns nothing. Split out so both timing modes agree."""
    for kind, side, price, qty in ops:
        if kind == "limit":
            live.append(book.limit(side, price, qty))
        elif kind == "market":
            book.market(side, qty)
        elif kind == "cancel_id":
            book.cancel(qty)          # qty carries the order id in file mode
        elif live:
            book.cancel(live.pop() if len(live) % 2 else live.pop(0))


def timer_overhead(samples=20_000):
    """What the clock itself costs, so it can be subtracted mentally."""
    t = []
    for _ in range(samples):
        a = time.perf_counter_ns()
        b = time.perf_counter_ns()
        t.append(b - a)
    t.sort()
    return t[len(t) // 2]


def bench_throughput(n, seed=42, ops=None):
    ops = ops if ops is not None else make_stream(n, seed)
    n = len(ops)
    book, live = Book(), []
    start = time.perf_counter_ns()
    _run(book, ops, live)
    elapsed = time.perf_counter_ns() - start
    return {
        "ops": n,
        "seconds": elapsed / 1e9,
        "ops_per_sec": n / (elapsed / 1e9),
        "ns_per_op": elapsed / n,
        "events": len(book.events),
    }


def bench_latency(n, seed=42, ops=None):
    ops = ops if ops is not None else make_stream(n, seed)
    book, live, lat = Book(), [], []
    for kind, side, price, qty in ops:
        t0 = time.perf_counter_ns()
        if kind == "limit":
            oid = book.limit(side, price, qty)
            live.append(oid)
        elif kind == "market":
            book.market(side, qty)
        elif kind == "cancel_id":
            book.cancel(qty)
        elif live:
            book.cancel(live.pop())
        else:
            continue
        lat.append(time.perf_counter_ns() - t0)
    lat.sort()

    def pct(p):
        return lat[min(int(len(lat) * p), len(lat) - 1)]

    return {"n": len(lat), "p50": pct(0.50), "p90": pct(0.90), "p99": pct(0.99),
            "p999": pct(0.999), "max": lat[-1]}


def histogram(n, seed=42, buckets=6):
    """Coarse log-ish histogram — where the tail actually lives."""
    ops = make_stream(n, seed)
    book, live, lat = Book(), [], []
    for kind, side, price, qty in ops:
        t0 = time.perf_counter_ns()
        if kind == "limit":
            live.append(book.limit(side, price, qty))
        elif kind == "market":
            book.market(side, qty)
        elif live:
            book.cancel(live.pop())
        else:
            continue
        lat.append(time.perf_counter_ns() - t0)
    edges = [0, 500, 1_000, 2_000, 5_000, 20_000, float("inf")][:buckets + 1]
    rows = []
    for lo, hi in zip(edges, edges[1:]):
        c = sum(1 for x in lat if lo <= x < hi)
        rows.append((lo, hi, c, 100 * c / len(lat)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--warmup", type=int, default=20_000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ops", help="replay a shared op file instead of generating a stream")
    a = ap.parse_args()
    n = 20_000 if a.quick else a.n
    warm = 5_000 if a.quick else a.warmup

    print(f"warmup {warm:,} ops (discarded)")
    bench_throughput(warm, seed=1)

    oh = timer_overhead(2_000 if a.quick else 20_000)
    print(f"timer overhead (median): {oh} ns — treat anything near this as noise\n")

    file_ops = load_ops(a.ops) if a.ops else None
    if file_ops:
        n = len(file_ops)
        print(f"workload: op file {a.ops} ({n:,} ops)\n")
    t = bench_throughput(n, ops=file_ops)
    print(f"THROUGHPUT  {t['ops']:,} ops in {t['seconds']:.3f}s")
    print(f"            {t['ops_per_sec']:,.0f} ops/sec  ({t['ns_per_op']:.0f} ns/op)")
    print(f"            {t['events']:,} events emitted\n")

    l = bench_latency(n, ops=file_ops)
    print(f"LATENCY     per-op, timed individually (n={l['n']:,})")
    for k in ("p50", "p90", "p99", "p999", "max"):
        print(f"            {k:>5}: {l[k]:>9,} ns")
    print()

    print("DISTRIBUTION")
    for lo, hi, c, pctg in histogram(n):
        label = f"{lo:,}-{hi:,}ns" if hi != float("inf") else f">{lo:,}ns"
        print(f"            {label:>18}  {c:>8,}  {pctg:5.1f}%  {'#' * int(pctg / 2)}")

    print("\nnote: pure Python, single thread. This is the BASELINE the compiled\n"
          "      rewrite gets measured against on the same stream and seed.")


if __name__ == "__main__":
    main()
