# Bullpen Engine

A limit order book exchange simulator and microstructure research framework, built from scratch.

Four pieces, each usable on its own:

1. **Matching engine** with correct price-time priority
2. **Live capture** of real exchange order book data
3. **Replay and fill simulation** that models queue position instead of pretending you fill at mid
4. **Research pipeline** with purged walk-forward validation

The Python side is standard library plus numpy; the Rust port is standard library only. No API keys, no paid data, no framework.

## Quick start

```bash
python test.py                    # run every self-check
python bench.py                   # benchmark the engine
python bench.py --quick           # faster pass

pip install websockets            # only needed for live capture
python capture.py --live BTCUSDT 300   # record 5 minutes of real book data
```

Each module also runs its own checks directly: `python orderbook.py`, `python capture.py`, `python replay.py`, `python research.py`.

## Benchmarks

20,000 mixed operations (75% limit, 10% market, 15% cancel), fixed seed, single thread, pure Python.

| | before | after | change |
|---|---|---|---|
| throughput | 63,311 ops/sec | 269,061 ops/sec | 4.25x |
| p50 latency | 3,376 ns | 2,505 ns | 1.35x |
| p99 latency | 354,360 ns | 12,644 ns | 28x |
| p99.9 latency | 771,027 ns | 64,270 ns | 12x |
| ops over 20us | 13.2% | 0.9% | |

**Methodology.** Warmup runs are discarded. Throughput is measured as total wall time over N operations, so no per-operation timer overhead contaminates it. Latency percentiles are measured per operation with `perf_counter_ns`, whose median overhead is about 100ns on this machine and is reported alongside the results, so anything near that floor is measuring the clock rather than the engine. The order stream is generated from a fixed seed, so a future compiled rewrite can be compared on identical input.

## Rust vs Python

Both engines replaying the **same 200,000-op file**, single thread, one machine, one session. Both emitted 239,703 events, which is the check that they did identical work: a ratio measured on two separately generated streams is not a result.

| | Python | Rust | ratio |
|---|---|---|---|
| throughput | 267,271 ops/sec | 3,398,667 ops/sec | 12.7x |
| ns per op | 3,742 | 294 | 12.7x |
| p90 | 3,500 ns | 400 ns | 8.8x |
| p99 | 8,900 ns | 1,200 ns | 7.4x |
| p99.9 | 80,800 ns | 6,100 ns | 13.2x |
| worst case | 31.6 ms | 1.9 ms | 16.4x |

Correctness is established separately and first: `diff_test.py` feeds both engines the same stream and compares event logs line by line. They match exactly, 24,065 events on a 20,000-op file. Speed without that check is worthless.

**Reading the p50 honestly.** On Windows the timer reports 0ns between two back-to-back calls, because QueryPerformanceCounter ticks at roughly 100ns. The Rust p50 of 200ns is therefore about two clock ticks, at the measurement floor rather than a precise figure. The throughput number does not use a per-op timer and independently agrees (294 ns/op), so quote throughput and p99, and say so if asked about the median. The binary prints this warning itself whenever the measured timer overhead is 0.

**Where the tail comes from.** Both engines still show a multi-millisecond worst case, and neither is a bug in the matching logic: it is the allocator growing the event log and, in Python, the garbage collector. That is why p99.9 is quoted rather than the max, and why the max is shown at all instead of hidden.

**How the Python speedup happened**, because the process matters more than the number. The first attempt replaced `max()`/`min()` scans over the price dictionaries with a bisect-maintained sorted index. That improved the median by 24% and barely moved the tail, which meant the hypothesis was wrong. Profiling per operation type showed why: cancels had an 82us median against 2.4us for limits, 34x worse, because removing an order from the middle of its queue is a linear scan and `deque` makes random access worse. GC was ruled out by rerunning with collection disabled. Replacing eager removal with lazy cancellation, where a cancel flags the order and decrements a per-level live counter, produced the numbers above.

## Design decisions

**Integer tick prices, never floats.** `0.1 + 0.2 != 0.3`. Float money bugs are silent and unrecoverable. Conversion happens only at the edges.

**Lazy cancellation with per-level live counts.** Cancelling flags the order rather than unlinking it, and matching skips the husks. The live counter is not optional: without it, a level holding nothing but cancelled orders would keep advertising a price and the book would quote a level that cannot trade.

**Fills at the resting order's price.** The order that was already there set the terms. Filling an aggressive buy at its own limit price instead of the resting ask silently invents profit.

**Trades captured alongside depth.** Depth updates alone cannot distinguish a trade from a cancel, and that distinction is the entire basis of queue-position modelling.

**Pessimistic queue model.** You join behind everything visible at your price. Trades consume the front of the queue and advance you. Cancels are assumed to come from behind you, so they never help. This under-fills rather than inventing fills, which is the correct direction to be wrong in.

**Purged walk-forward validation.** A label that looks forward `h` steps means a training row at index `i` has already seen the future if `i + h` reaches into the test block. Standard k-fold does this constantly and inflates every metric. Those rows are dropped, plus an embargo window after the test block.

## What is not built yet

Being explicit, since a half-built project described honestly is worth more than a finished one described loosely:

- **Rust engine is built and verified** (see above); what remains is multi-symbol support and a live-capture run against real Binance traffic.
- **Gradient-boosted trees.** Only logistic regression is implemented, deliberately, as a baseline. Trees need sklearn or lightgbm.
- **Parquet storage.** Capture writes gzipped JSONL. Columnar storage earns its place at analysis scale, not at capture time.
- **Multi-symbol support.** One book, one symbol.
- **A live-capture run against real Binance traffic.** The sequencing state machine is covered by tests; the socket path itself has only been exercised against synthetic data.

## An honest note on results

This project is infrastructure and methodology. It makes no claim of a profitable trading strategy, and the evaluation is deliberately built to make weak edges visible rather than hide them: costs are charged on every trade, and a 1 basis point cost is enough to erase a half basis point edge. Most microstructure signals are smaller than the spread, so a negative net PnL after costs is the normal and expected outcome, not a bug to tune away.

The claim worth making is the rigor: no lookahead, purged splits, honest benchmarks with documented methodology, and a fill model that under-promises.

## Layout

```
orderbook.py   matching engine, price-time priority, lazy cancels
capture.py     Binance L2 capture, snapshot/diff resequencing, gap detection
replay.py      book reconstruction, queue-position fills, microstructure features
research.py    labels, purged walk-forward CV, baseline model, cost-aware metrics
bench.py       throughput and tail latency harness
diff_test.py   proves the Rust engine matches this one, event for event
test.py        runs every self-check
rust/          the compiled port: same semantics, 12.7x throughput
```
