"""
metrics.py — latency histograms and counters.

A trading system that cannot answer "what was p99 in the last hour" is not
operable. The two things worth measuring are counts (orders, fills, rejects,
gaps) and latency distributions, and the second is where naive code goes wrong
in a specific way: averaging.

A mean latency hides exactly the event you care about. An engine averaging
200ns that stalls for 4ms during a burst looks identical to one that never
leaves 250ns, and the stall is the one that costs money, because the slow path
runs when the market is moving. So this records a distribution, not a running
average.

HOW: a log-linear bucketed histogram, which is what HdrHistogram and
Prometheus both do. Buckets are fixed, so recording is O(1) with no allocation
and no sample retention; memory is constant regardless of how many values go
in. The cost is that percentiles are approximate, bounded by bucket width, and
that bound is reported rather than glossed over.

The alternative, keeping every sample and sorting, is exact but allocates
without limit and cannot run in a hot path for an entire session.

Run `python metrics.py` for the self-check.
"""

import math


class Histogram:
    """Log-linear bucketed latency histogram, fixed memory.

    `precision` buckets per power of two. Higher is more accurate and uses
    more memory; 8 keeps relative error under about 1/16.
    """

    def __init__(self, precision=8):
        if precision < 2:
            raise ValueError("precision must be at least 2")
        self.precision = precision
        self.counts: dict[int, int] = {}
        self.count = 0
        self.total = 0
        self.min = None
        self.max = None

    def _bucket(self, value: int) -> int:
        if value <= 0:
            return 0
        exp = value.bit_length() - 1              # floor(log2(value))
        # position within this power of two, split into `precision` sub-buckets
        span = 1 << exp
        sub = ((value - span) * self.precision) // span
        return exp * self.precision + sub + 1

    def _bucket_low(self, idx: int) -> float:
        if idx == 0:
            return 0.0
        idx -= 1
        exp, sub = divmod(idx, self.precision)
        span = 1 << exp
        return span + (span * sub) / self.precision

    def _bucket_high(self, idx: int) -> float:
        if idx == 0:
            return 1.0
        idx -= 1
        exp, sub = divmod(idx, self.precision)
        span = 1 << exp
        return span + (span * (sub + 1)) / self.precision

    def record(self, value):
        value = int(value)
        if value < 0:
            raise ValueError("cannot record a negative latency")
        self.counts[self._bucket(value)] = self.counts.get(self._bucket(value), 0) + 1
        self.count += 1
        self.total += value
        self.min = value if self.min is None else min(self.min, value)
        self.max = value if self.max is None else max(self.max, value)

    def percentile(self, p: float) -> float:
        """Approximate percentile. Returns the bucket's upper bound.

        Upper bound rather than midpoint, so a reported p99 is never optimistic.
        """
        # Validate before the empty-histogram shortcut, or a bad percentile
        # silently returns 0.0 instead of raising.
        if not 0 <= p <= 100:
            raise ValueError("percentile must be between 0 and 100")
        if not self.count:
            return 0.0
        target = math.ceil(self.count * p / 100.0)
        target = max(1, min(target, self.count))
        seen = 0
        for idx in sorted(self.counts):
            seen += self.counts[idx]
            if seen >= target:
                return min(self._bucket_high(idx), float(self.max))
        return float(self.max)

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    def error_bound(self) -> float:
        """Worst-case relative error from bucketing, as a fraction."""
        return 1.0 / self.precision

    def snapshot(self):
        return {
            "count": self.count,
            "min": self.min or 0,
            "mean": round(self.mean, 1),
            "p50": self.percentile(50),
            "p90": self.percentile(90),
            "p99": self.percentile(99),
            "p99.9": self.percentile(99.9),
            "max": self.max or 0,
            "rel_error": self.error_bound(),
        }


class Metrics:
    """Named counters and histograms, with a flat snapshot for scraping."""

    def __init__(self):
        self.counters: dict[str, int] = {}
        self.histograms: dict[str, Histogram] = {}

    def incr(self, name, by=1):
        self.counters[name] = self.counters.get(name, 0) + by
        return self.counters[name]

    def observe(self, name, value):
        h = self.histograms.get(name)
        if h is None:
            h = self.histograms[name] = Histogram()
        h.record(value)
        return h

    def timer(self, name):
        """`with metrics.timer('match'):` records elapsed nanoseconds."""
        return _Timer(self, name)

    def snapshot(self):
        return {"counters": dict(self.counters),
                "latency": {k: h.snapshot() for k, h in self.histograms.items()}}

    def render_prometheus(self) -> str:
        """Text exposition format, so this can be scraped as-is."""
        lines = []
        for name, value in sorted(self.counters.items()):
            safe = name.replace(".", "_").replace("-", "_")
            lines.append(f"# TYPE {safe} counter")
            lines.append(f"{safe} {value}")
        for name, h in sorted(self.histograms.items()):
            safe = name.replace(".", "_").replace("-", "_")
            lines.append(f"# TYPE {safe}_nanoseconds summary")
            for q, key in (("0.5", "p50"), ("0.9", "p90"),
                           ("0.99", "p99"), ("0.999", "p99.9")):
                lines.append(f'{safe}_nanoseconds{{quantile="{q}"}} '
                             f'{h.snapshot()[key]}')
            lines.append(f"{safe}_nanoseconds_count {h.count}")
            lines.append(f"{safe}_nanoseconds_sum {h.total}")
        return "\n".join(lines) + "\n"


class _Timer:
    def __init__(self, metrics, name):
        self.metrics, self.name = metrics, name

    def __enter__(self):
        import time
        self._t0 = time.perf_counter_ns()
        return self

    def __exit__(self, *exc):
        import time
        self.metrics.observe(self.name, time.perf_counter_ns() - self._t0)
        return False


def demo():
    # ---- bucketing is monotonic and brackets the value ----
    h = Histogram(precision=8)
    for v in (1, 2, 3, 100, 1000, 10_000, 1_000_000):
        b = h._bucket(v)
        assert h._bucket_low(b) <= v <= h._bucket_high(b), (v, b)
    assert h._bucket(10) <= h._bucket(100) <= h._bucket(1000), "must be ordered"

    # ---- percentiles land within the stated error bound ----
    h = Histogram(precision=8)
    for v in range(1, 10_001):
        h.record(v)
    assert h.count == 10_000 and h.min == 1 and h.max == 10_000
    for p, exact in ((50, 5000), (90, 9000), (99, 9900)):
        got = h.percentile(p)
        err = abs(got - exact) / exact
        assert err <= h.error_bound(), f"p{p}: {got} vs {exact}, err {err:.3f}"
    assert abs(h.mean - 5000.5) < 1e-6, "mean is exact, it is not bucketed"

    # never optimistic: the reported percentile is the bucket's upper edge
    h2 = Histogram()
    for _ in range(100):
        h2.record(1000)
    assert h2.percentile(99) >= 1000

    # ---- the case a mean hides ----
    fast = Histogram()
    bursty = Histogram()
    for _ in range(9_999):
        fast.record(250)
        bursty.record(200)
    fast.record(250)
    bursty.record(4_000_000)                    # one 4ms stall
    # One stall in 10,000 moves the mean by about 2x and the tail by 16,000x.
    # A single outlier sits at p100, not p99.99, which is itself the lesson:
    # pick the percentile for the event rate you actually care about.
    mean_ratio = bursty.mean / fast.mean
    tail_ratio = bursty.max / fast.max
    assert mean_ratio < 3, f"mean moved {mean_ratio:.1f}x"
    assert tail_ratio > 1_000, f"tail only moved {tail_ratio:.0f}x"
    assert bursty.percentile(99.99) <= 2 * fast.percentile(99.99), \
        "p99.99 of 10k samples cannot see a single outlier"
    assert bursty.max == 4_000_000

    # ---- edges ----
    empty = Histogram()
    assert empty.percentile(50) == 0.0 and empty.snapshot()["count"] == 0
    z = Histogram(); z.record(0)
    assert z.min == 0 and z.percentile(50) >= 0
    for bad in (-1,):
        try:
            Histogram().record(bad); raise AssertionError("recorded a negative")
        except ValueError:
            pass
    try:
        Histogram().percentile(101); raise AssertionError("accepted p>100")
    except ValueError:
        pass
    try:
        Histogram(precision=1); raise AssertionError("accepted precision 1")
    except ValueError:
        pass

    # memory stays bounded no matter how many samples arrive
    big = Histogram()
    for v in range(1, 200_000):
        big.record(v)
    assert len(big.counts) < 300, f"bucket count grew to {len(big.counts)}"

    # ---- counters, timers, exposition ----
    m = Metrics()
    m.incr("orders.accepted")
    m.incr("orders.accepted", 4)
    m.incr("orders.rejected")
    assert m.counters["orders.accepted"] == 5
    for v in (100, 200, 300, 10_000):
        m.observe("match.latency", v)
    with m.timer("block"):
        sum(range(1000))
    snap = m.snapshot()
    assert snap["counters"]["orders.accepted"] == 5
    assert snap["latency"]["match.latency"]["count"] == 4
    assert snap["latency"]["block"]["count"] == 1
    assert snap["latency"]["match.latency"]["max"] == 10_000

    text = m.render_prometheus()
    assert "orders_accepted 5" in text
    assert 'match_latency_nanoseconds{quantile="0.99"}' in text
    assert "match_latency_nanoseconds_count 4" in text
    assert text.endswith("\n")

    print(f"ok — log-linear histogram (rel err <= {h.error_bound():.3f}), "
          f"bounded memory, counters, Prometheus exposition")


if __name__ == "__main__":
    demo()
