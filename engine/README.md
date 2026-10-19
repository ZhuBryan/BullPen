# Bullpen Engine

A limit order book exchange simulator and microstructure research framework, built from scratch.

The pieces, each usable on its own:

1. **Matching engine** with correct price-time priority, in Python, Rust and C++17 (all three differential-tested)
2. **FIX 4.4 order entry over TCP**, with a real session layer, feeding an OMS with cancel/replace and average-cost PnL
3. **Pre-trade risk**: size, price collars, position limits, self-trade prevention
4. **Instrument reference data** and multi-symbol routing (Reg NMS tick regimes)
5. **Live capture** of real exchange order book data from Kraken (checksum-verified) and Binance (sequence-verified)
6. **Replay and fill simulation** that models queue position instead of pretending you fill at mid
7. **Execution algorithms** (TWAP, VWAP, POV) with implementation-shortfall TCA
8. **Research pipeline** with purged walk-forward validation
9. **Metrics**: latency histograms and counters, Prometheus exposition
10. **Python bindings for the Rust engine** (PyO3), so the rest of the project can run on it

The Python side is standard library plus numpy; the Rust port is standard library only. No API keys, no paid data, no framework.

## Quick start

```bash
python test.py                    # run every self-check (incl. live capture)
python bench.py                   # benchmark the engine (5 repeats, median and spread)
python bench.py --quick           # faster pass
python bench.py --repeats 9 --json out.json   # machine-readable, as CI archives it

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

**Methodology.** Each figure is the median of repeated runs (5 by default), reported with its min-max range and interquartile range, so a single lucky run can't be quoted. Any latency percentile within twice the measured timer overhead is flagged "at timer floor, do not quote". Warmup runs are discarded. Throughput is measured as total wall time over N operations, so no per-operation timer overhead contaminates it. Latency percentiles are measured per operation with `perf_counter_ns`, whose median overhead is about 100ns on this machine and is reported alongside the results, so anything near that floor is measuring the clock rather than the engine. The order stream is generated from a fixed seed, so a future compiled rewrite can be compared on identical input.

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

**These numbers predate event timestamping.** Every event now carries a monotonic sequence number and a nanosecond timestamp, which costs **16.5% of throughput in Python** (267,271 down to 223,025 ops/sec on the same 200,000-op file), of which about 6 points is the clock read itself and the rest is the wider event tuple. That is the honest price of a feed that looks like a real venue's, and it is why the clock is injectable: pass a counter and you get the structure without the syscall. Re-run both benchmarks to refresh the table above.

**Reading the p50 honestly.** On Windows the timer reports 0ns between two back-to-back calls, because QueryPerformanceCounter ticks at roughly 100ns. The Rust p50 of 200ns is therefore about two clock ticks, at the measurement floor rather than a precise figure. The throughput number does not use a per-op timer and independently agrees (294 ns/op), so quote throughput and p99, and say so if asked about the median. The binary prints this warning itself whenever the measured timer overhead is 0.

**Where the tail comes from.** Both engines still show a multi-millisecond worst case, and neither is a bug in the matching logic: it is the allocator growing the event log and, in Python, the garbage collector. That is why p99.9 is quoted rather than the max, and why the max is shown at all instead of hidden.

**How the Python speedup happened**, because the process matters more than the number. The first attempt replaced `max()`/`min()` scans over the price dictionaries with a bisect-maintained sorted index. That improved the median by 24% and barely moved the tail, which meant the hypothesis was wrong. Profiling per operation type showed why: cancels had an 82us median against 2.4us for limits, 34x worse, because removing an order from the middle of its queue is a linear scan and `deque` makes random access worse. GC was ruled out by rerunning with collection disabled. Replacing eager removal with lazy cancellation, where a cancel flags the order and decrements a per-level live counter, produced the numbers above.

## Concurrency

A real engine runs the feed handler and the matcher on separate threads, and the handoff is where the engineering lives. `rust/src/spsc.rs` is a lock-free single-producer single-consumer ring buffer:

- `head` is producer-owned, `tail` consumer-owned, so each side loads its own index `Relaxed` and the other's `Acquire`. The split is enforced by the type system: `channel()` returns two separate handles, so a second producer is not constructible.
- The producer writes the slot, then publishes `head` with `Release`; the consumer reads `head` with `Acquire`. That pairing is what guarantees the slot write is visible before the index advertising it. With `Relaxed` the consumer can legally read uninitialized memory, and it will still pass every test on x86 before failing on ARM.
- `head` and `tail` are padded onto separate 64-byte cache lines. Without that, two threads touching no shared data still ping-pong a cache line between cores.
- Indices increase monotonically and are masked to a slot, which removes the full-versus-empty ambiguity without sacrificing a slot.

Backpressure returns the value to the caller instead of blocking or dropping it: a feed that silently discards messages is worse than one that reports being behind.

```bash
cargo run --release -- pipeline ../ops.txt
```

That mode runs the same op stream through the queue and through a `Mutex<VecDeque>` baseline, asserts both produced identical event counts, and prints the ratio. Comparing against a mutex on identical work is the honest way to show whether lock-free earns its complexity.

Measured on a 20,000-op file (single run, Windows, release build):

| Handoff | Throughput | Events |
|---|---|---|
| Lock-free SPSC | 2.50M ops/sec | 24,065 |
| `Mutex<VecDeque>` | 2.10M ops/sec | 24,065 |

That's a 1.19x gain, which is smaller than you might expect, and the backpressure counter explains why. The producer hit a full queue 132,821 times while pushing 20,000 ops, about 6.6 retries per op. So the feed thread spent most of its time waiting for the matcher, not fighting over a lock. The queue isn't the bottleneck at this rate, so 1.19x is a floor on the difference rather than its size. It would widen with a cheaper consumer or a burstier producer. The run is also too short (8ms) to quote with any precision, so I'm reporting it as measured rather than as a headline.

## Two ways to keep a book honest: Binance and Kraken

Kraken is the second live source, added because Binance is unavailable from Canada and the US. `python capture.py --live kraken BTC/USD 3600` subscribes to Kraken's WebSocket v2 book (depth 10) and trade channels and writes the same gzip JSONL records as the Binance path, so replay and research work on either.

The two exchanges protect book integrity in opposite ways. Binance numbers every update, so a missing number tells you that you missed a message *before* you apply anything, and you resync from a REST snapshot. Kraken v2 sends no sequence numbers. Instead each message carries a CRC32 checksum of the top 10 levels as they should look after the update, so you apply it, hash your own book and compare. A mismatch tells you your state is wrong whatever the cause (a lost message, a parsing bug, a forgotten depth truncation), and the fix is to resubscribe for a fresh snapshot. The checksum catches more kinds of bug than a sequence number, but only after the fact and only for the top 10 levels. Prices and sizes are held as exact integer ticks and lots, which is also what makes the checksum string trivial to build, and the implementation is pinned to the worked example in Kraken's own documentation (checksum 3310070434).

`research.py` then runs an order-flow forecast on a capture: OFI over a window, top-of-book imbalance and spread at time t, predicting the sign of the mid move over the next h events, with purged walk-forward folds, an overlap-adjusted IC t-stat, baselines, costs and negative controls. One finding worth knowing: the older microprice-label pipeline scores AUC around 0.65 on pure noise, because queue sizes mean-revert, so use the mid-based evaluation for any real claim.

## Order entry, FIX and the OMS

`fix.py` speaks FIX 4.4, the protocol equity and futures order entry actually
runs on. BodyLength and a zero-padded CheckSum are both computed correctly and
pinned by tests, including rejection of tampered messages, and the order state
machine makes illegal transitions raise rather than letting a venue fill an
order the client already cancelled.

`oms.py` is the client-facing layer: ClOrdIDs, cancel and cancel/replace
addressed by OrigClOrdID, execution reports as the source of truth, and
average-cost position accounting. Two details that are easy to get wrong and
are tested here:

- **Replace loses queue priority.** Amending is a cancel plus a new order, and
  the replacement goes to the back of the queue at its price. A test asserts a
  later order fills first after a replace.
- **A position flipping through zero** closes the old side and opens the
  remainder at the fill price. Blending them into one average misstates both
  realized and unrealized PnL.

### The FIX session layer

`fix_session.py` is the part that makes FIX usable over a real socket. TCP is a byte stream, so messages are framed by reading `8=FIX.4.4`, then the BodyLength, then exactly that many bytes, then the checksum trailer. A garbled message is discarded without consuming a sequence number.

Every message carries a sequence number in both directions. When an incoming number skips ahead, the receiver sends a ResendRequest for the gap and holds what arrived early. The sender replays business messages with PossDupFlag=Y and their original sending time, and replaces runs of admin messages (heartbeats, test requests) with a single SequenceReset-GapFill. Replaying a stale heartbeat is pointless; replaying a stale order is dangerous, which is why the duplicate flag exists, and why the OMS rejecting a duplicate ClOrdID is what makes replays safe. A number that's too low *without* the duplicate flag is a serious error and ends the session. Idle sides heartbeat, a silent peer gets a TestRequest, and no answer means disconnect.

The acceptor routes NewOrderSingle, cancel and cancel/replace into the OMS and sends execution reports back to whichever session owns the order. Review found real holes here, all now closed and regression-tested: a rejected duplicate ClOrdID used to steal ownership of the original (letting one client cancel another's order and receive its reports), a ResendRequest with a negative start froze every session on the shared event loop, a second Logon for a connected CompID hijacked the live session, and malformed fields killed the read loop.

## Execution algorithms and TCA

`execution.py` has TWAP, VWAP (largest-remainder allocation, so slices conserve
shares) and POV. `backend/exec_sim.py` works a parent order through the live
simulated book, so child orders pay the spread and move the market rather than
filling at an assumed price.

Scoring is implementation shortfall against the arrival price, decomposed into
the cost of what traded and the opportunity cost of what did not. That second
term is the point: an algorithm that quietly fills 60% and reports a beautiful
average price has hidden its largest cost. The sign convention is pinned both
ways in tests: positive always means worse for the trader, whether buying or
selling.

## Pre-trade risk

`risk.py` is the gate in front of the engine, which is where venues put it: risk rules change per account and often, matching logic must not. Five checks, all with recorded reject codes for audit:

| check | catches |
|---|---|
| max order size / value | the fat-finger order |
| price collar (bps from last trade) | orders far from the market that will be picked off |
| position limit | net exposure, enforced on fills rather than submissions |
| open order limit | a runaway algo flooding the book |
| self-trade prevention | an account matching its own resting order |

A test asserts the gate never changes *how* orders match, only whether they are admitted.

## Design decisions

**Integer tick prices, never floats.** `0.1 + 0.2 != 0.3`. Float money bugs are silent and unrecoverable. Conversion happens only at the edges.

**Sequence numbers are the ordering, not timestamps.** Every event carries both, the way an ITCH feed does, but `seq` is authoritative: it is dense, monotonic, and reproducible across a replay, while a clock reading is process-local and can be identical for two events on a coarse timer. So the differential test compares sequence numbers and deliberately ignores timestamps, since two engines can never agree on a wall clock and failing on that would say nothing about whether the matching logic agrees. The clock is injectable for the same reason: `Book(clock=counter_clock())` in Python and `Book::deterministic()` in Rust make a log byte-reproducible.

Event payloads:

| event | fields (before `seq`, `ts_ns`) |
|---|---|
| `accepted` | id, side, price (`M` for market), qty |
| `trade` | price, qty, aggressor id, resting id |
| `canceled` | id, remaining |
| `reject_cancel` | id |

**Lazy cancellation with per-level live counts.** Cancelling flags the order rather than unlinking it, and matching skips the husks. The live counter is not optional: without it, a level holding nothing but cancelled orders would keep advertising a price and the book would quote a level that cannot trade.

**Fills at the resting order's price.** The order that was already there set the terms. Filling an aggressive buy at its own limit price instead of the resting ask silently invents profit.

**Trades captured alongside depth.** Depth updates alone cannot distinguish a trade from a cancel, and that distinction is the entire basis of queue-position modelling.

**Pessimistic queue model.** You join behind everything visible at your price. Trades consume the front of the queue and advance you. Cancels are assumed to come from behind you, so they never help. This under-fills rather than inventing fills, which is the correct direction to be wrong in.

**Purged walk-forward validation.** A label that looks forward `h` steps means a training row at index `i` has already seen the future if `i + h` reaches into the test block. Standard k-fold does this constantly and inflates every metric. Those rows are dropped, plus an embargo window after the test block.

## What is not built yet

Being explicit, since a half-built project described honestly is worth more than a finished one described loosely:

- **No C++.** Rust covers the systems ground, but most quant job specs ask for C++. The differential harness would validate a third port for free.
- **The matching engine itself is still single-threaded.** The lock-free queue and the two-thread pipeline exist and are tested, but the deterministic single-threaded path is what the differential test runs against, and that is deliberate: the oracle has to stay reproducible.
- **Gradient-boosted trees.** Only logistic regression is implemented, deliberately, as a baseline. Trees need sklearn or lightgbm.
- **No persistent FIX message store.** Sequence numbers and the resend store live in memory, so they don't survive a restart, and execution reports for a logged-out client are dropped rather than queued. A production engine would persist both, plus TLS and scheduled sessions.
- **Parquet storage.** Capture writes gzipped JSONL. Columnar storage earns its place at analysis scale, not at capture time.
- **A capture run against *real* Binance traffic.** The live socket path is now tested end to end, but against a local fake feed (`test_capture_live.py`), not the real exchange. That test stands up an actual websocket server and HTTP snapshot endpoint on localhost, scripts a deliberate sequence gap, and asserts the gap is caught, a resync snapshot follows, the trade stream is routed, a dropped socket is survived, and the gzip file round-trips. It found a real bug on first run: a closed socket raised an unhandled `ConnectionClosedOK` and killed the recorder, so `record()` now reconnects with exponential backoff and re-snapshots on every reconnect (mandatory, since the book moved while the socket was down). What remains untested is the real feed's quirks: rate limits, 24-hour connection recycling, and genuinely out-of-order traffic.

## An honest note on results

This project is infrastructure and methodology. It makes no claim of a profitable trading strategy, and the evaluation is deliberately built to make weak edges visible rather than hide them: costs are charged on every trade, and a 1 basis point cost is enough to erase a half basis point edge. Most microstructure signals are smaller than the spread, so a negative net PnL after costs is the normal and expected outcome, not a bug to tune away.

The claim worth making is the rigor: no lookahead, purged splits, honest benchmarks with documented methodology, and a fill model that under-promises.

## Layout

```
orderbook.py   matching engine, price-time priority, lazy cancels
capture.py     Binance L2 capture, snapshot/diff resequencing, gap detection
replay.py      book reconstruction, queue-position fills, microstructure features
research.py    labels, purged walk-forward CV, baseline model, cost-aware metrics
risk.py        pre-trade checks: size, value, price collar, position, self-trade
fix.py         FIX 4.4 encode/decode, builders, order state machine
fix_session.py FIX session layer over TCP: logon, heartbeats, gap recovery, acceptor -> OMS
oms.py         order lifecycle, cancel/replace, average-cost positions and PnL
execution.py   TWAP/VWAP/POV scheduling and transaction cost analysis
instruments.py tick regimes, lot rules, multi-symbol Venue
metrics.py     latency histograms, counters, Prometheus text
bench.py       throughput and tail latency harness
diff_test.py   proves the Rust engine matches this one, event for event
fastbook.py    picks the Rust book when its Python extension is built, else this one
test_pyo3.py   20,000 random ops through both engines via PyO3, every event compared
test.py        runs every self-check
test_capture_live.py  live capture against a local fake Binance feed
rust/          the compiled port: same semantics, 12.7x throughput
rust/src/spsc.rs  lock-free SPSC queue for the feed -> matcher handoff
rust/src/py.rs    PyO3 binding (built only with the `python` feature)
cpp/              C++17 port: header-only engine, replay/bench/selftest, Makefile, MSVC-compatible
bench_sim.py      end-to-end simulation timing, Python vs Rust; --profile measures the book's true share
rust/bench/       criterion benchmarks, a separate crate so the engine stays dependency-free
```
