# I guessed wrong about where my order book was slow

I built a limit order book matching engine, optimized it, and then ported it to Rust. The headline numbers are a 4.2x throughput gain in Python, a 28x improvement in p99 latency, and another 12.7x from the port.

Those numbers are not the interesting part. The interesting part is that my first optimization was based on a confident hypothesis that turned out to be wrong, and I only found out because I measured the thing I had actually changed instead of the thing I expected to improve.

This writeup is mostly about the wrong guesses. There were four.

## What the engine does

A limit order book is the ledger an exchange keeps of every resting order. Buyers sit on one side sorted high to low, sellers on the other sorted low to high, and when a new order crosses the gap between them, the engine pairs it against whatever is resting and emits trades.

The matching rule is price-time priority, which is what real venues use:

1. A better price wins.
2. At the same price, whoever arrived first wins.

That second rule is what makes the data structure interesting. Each price level is a FIFO queue, so the engine is not one sorted container but a sorted collection of queues.

Two decisions up front. Prices are **integer ticks, never floats**, because `0.1 + 0.2 != 0.3` and money bugs from float rounding are silent and unrecoverable. And every action emits an event (`accepted`, `trade`, `canceled`, `reject_cancel`) carrying a monotonic sequence number, which is what makes replay deterministic later.

## Rule one: correctness before speed, and prove it

Before any optimization, the engine had a self-check asserting the invariants that must hold after every single operation:

- the book is never crossed (best bid is always below best ask)
- FIFO order is preserved within a price level
- no price level is advertised with nothing live in it
- shares are conserved: `2 * filled + canceled + resting == submitted`

That last one needs the factor of two, which I got wrong on the first attempt. Every fill consumes quantity from *both* the aggressor and the resting order, so a trade of size `q` accounts for `2q` of the shares submitted. My assertion failed, I assumed the engine was leaking shares, and the engine was fine. The arithmetic in the test was wrong.

These run after every operation across 3,000 randomized operations, which is slow and completely worth it. An invariant you only check at the end tells you something broke; an invariant you check every step tells you *which operation* broke it.

## Wrong guess #1: the price-level scan

Here is the baseline, 20,000 mixed operations (75% limit, 10% market, 15% cancel):

```
throughput    63,311 ops/sec
p50            3,376 ns
p99          354,360 ns
p99.9        771,027 ns
ops > 20us        13.2%
```

A p99 of 354 microseconds against a median of 3.4 is a bad shape. In trading, the tail is the number that matters, because the slow path is exactly the path that runs when the market is moving and you most need to be fast. An engine that averages 200ns and stalls for 4ms on a resize is worse than one that never leaves 800ns.

My hypothesis was obvious and wrong. The code found the best bid with `max(self.bids)` over a dictionary of price levels. In Python that is an interpreted loop over every level, on every single lookup, and `_match` calls it repeatedly. So I replaced it with a `bisect`-maintained sorted list of active prices, making best bid a single index.

Result:

```
throughput    63,311 -> 71,996 ops/sec   (+14%)
p50            3,376 -> 2,574 ns         (-24%)
p99          354,360 -> 348,269 ns       (-2%)
ops > 20us      13.2% -> 13.0%
```

The median improved exactly as predicted. The tail did not move at all. Had I only reported throughput and p50 I would have written this up as a win and moved on with the real problem untouched.

## What the profiler actually said

A 2% tail improvement means the bottleneck is somewhere else entirely. So instead of guessing again, I measured latency separately per operation type:

```
            n        p50        p99         max
limit  15,053    2,434 ns   18,896 ns    917,655 ns
market  1,941    4,519 ns   30,126 ns    165,659 ns
cancel  3,006   82,414 ns  477,891 ns  1,916,789 ns
```

Cancels had a median of 82 microseconds against 2.4 for limits. **34x worse, on 15% of the traffic.**

I also ruled out the garbage collector, which was my second guess, by rerunning with collection disabled. The tail got slightly worse rather than better, so GC was not the cause.

The cause was one line. Cancelling looked up the order, then removed it from the middle of its price-level queue:

```python
level.remove(o)
```

That is a linear scan of the queue, and on a `deque` it is worse than linear-on-a-list because `deque` random access is itself O(n). On a busy price level with hundreds of resting orders, every cancel walked the whole thing.

I had actually flagged this in a code comment while writing it, with a note that an intrusive linked list was the upgrade if cancel-heavy workloads showed up in the profile. Then I guessed at the price scan anyway. The note was right and I did not believe it until the profiler said so.

## The fix: stop doing the removal

A doubly linked list with an ID-to-node map would make removal O(1). It is also what I did not do, because there is a lazier fix that real exchanges use: **do not remove the order at all.**

Cancelling now flags the order and decrements a counter:

```python
o.cancelled = True
live[o.price] -= 1
```

The flagged husk stays in the queue and is discarded when matching eventually reaches the front of it. Cancel becomes O(1), and the cleanup is amortized into work the engine was doing anyway.

The per-level live counter is not bookkeeping, it is load-bearing. Without it, a level holding nothing but cancelled husks would still appear in the price index, so the book would advertise a price that cannot possibly trade. The invariant check that catches this (`no level advertised with nothing live in it`) found it immediately when I first wrote the lazy path without the counter.

Results on the same stream:

```
throughput    63,311 -> 269,061 ops/sec   (4.25x)
p50            3,376 -> 2,505 ns
p99          354,360 -> 12,644 ns         (28x)
p99.9        771,027 -> 64,270 ns         (12x)
ops > 20us      13.2% -> 0.9%
```

The sorted price index stayed, because it genuinely helped the median. But it was not the fix, and reporting it as the fix would have been a lie that happened to come with real numbers attached.

## The Rust port, and proving it did not change behavior

A rewrite that is faster but subtly different is worse than no rewrite, because every result built on it is quietly wrong. So the Python engine became the oracle.

Both engines read the **same op file** and emit their event logs in the same wire format, which are then compared line by line. Generating a random stream separately in each language does not work, because two languages' RNGs will never agree. Order ids are assigned sequentially, so identical input yields identical ids and the logs compare directly.

```
$ python diff_test.py check ops.txt 20000
generated 20000 ops -> ops.txt
python: 24065 events
rust:   24065 events
IDENTICAL - 24065 events matched exactly.
```

Timestamps are deliberately excluded from that comparison. A clock reading is process-local by definition, so two engines can never agree on it, and diffing it would fail for a reason that says nothing about whether the matching logic agrees. Sequence numbers are compared instead: they are deterministic and they are the authoritative ordering anyway.

Only then, the speed. Both engines replaying the same 200,000-op file, single thread, same machine, same session:

| | Python | Rust | ratio |
|---|---|---|---|
| throughput | 267,271 ops/sec | 3,398,667 ops/sec | 12.7x |
| ns per op | 3,742 | 294 | 12.7x |
| p90 | 3,500 ns | 400 ns | 8.8x |
| p99 | 8,900 ns | 1,200 ns | 7.4x |
| p99.9 | 80,800 ns | 6,100 ns | 13.2x |
| worst case | 31.6 ms | 1.9 ms | 16.4x |

Both emitted 239,703 events, which is the check that they did identical work.

The data structures differ, and each difference exists because the language made a different choice cheap. Python needed the bisect workaround because `max(dict)` is an interpreted loop; Rust's `BTreeMap::last_key_value` is O(log n) and compiled, so no workaround is needed. Orders live in one flat `Vec` slab referenced by index rather than as separately allocated objects, which is the change that actually matters for cache behavior.

## Reading your own benchmark honestly

Three things that make a benchmark number mean something.

**Measure the timer.** The Rust harness reports the median cost of two back-to-back clock reads. On Windows it came back as **0 nanoseconds**, because `QueryPerformanceCounter` ticks at roughly 100ns and cannot resolve two adjacent calls. That makes the reported Rust p50 of 200ns about two clock ticks: at the measurement floor, not a precise figure. The throughput number does not use a per-operation timer and independently agrees at 294 ns/op. So the honest thing is to quote throughput and p99, and to say "that is near timer resolution" when asked about the median. The binary now prints this warning itself whenever measured overhead is zero.

**Measure two different things.** Throughput is wall time over N operations, with no per-operation timer in the loop, so timer cost cannot contaminate it. Percentiles need per-operation timing and therefore pay that cost. They answer different questions and both belong in the output.

**Feed both sides the same input.** My first Rust-versus-Python comparison had each engine generating its own stream from its own RNG. The workloads were statistically similar but not identical, which I noticed only because the event counts differed by 174. Statistically similar is not good enough to quote a ratio.

## What timestamping cost

Adding a sequence number and a nanosecond timestamp to every event, the way an ITCH feed is structured, cost **16.5% of Python throughput** (267,271 down to 223,025 ops/sec). About 6 points of that is the clock read itself; the rest was a wider event tuple and an extra function call, most of which I recovered by inlining the emit sites.

That cost is real and it is the price of a feed that looks like a real venue's. I mention it because the alternative, quietly leaving the old numbers in the README, was available and tempting.

## Wrong guesses #2 through #4

The order book was not the only place I was confidently wrong. Three more, all found the same way.

**A fill model that claimed to be pessimistic and was the opposite.** The replay harness models queue position: when you post at a price, you join behind everything already resting there, and you only advance when real trades consume the front. The function that measured "how much is ahead of me" looked the price up in the book by string key. But the book stores prices as the raw exchange strings, so `"68000.01000000"`, while the lookup built `"68000.01"`. It never matched, so it always returned zero, and zero means *front of the queue*: the single most optimistic assumption available. The module's documented claim of being deliberately pessimistic was exactly inverted, and no test caught it because every test passed `ahead=` by hand.

**A Sharpe ratio inflated by about 16%.** Sortino divides return by downside deviation, which is the root mean square of the negative returns taken over *every* observation. I had written `returns[returns < 0].std()`, which divides by the count of down days instead of the full sample and measures spread around the mean of the negatives instead of around zero. Both mistakes push the number the same direction, and they push it hardest for the strategies that already look best.

**A search that trained on the future.** A strategy search needs cross-validation that accounts for labels looking forward in time. Mine purged correctly on one side and then happily included everything *after* the test block in training: on the first fold, 76% of the training data came from the test period's future. The docstring said "train is always strictly before test." The code said otherwise.

The pattern in all four is the same. Each one was a confident claim in a comment or docstring that the code did not honor, and each was found by a check that compared the claim against measured behavior rather than against my intent.

## Honest limitations

The things this does not do, since a half-built project described accurately is worth more than a finished one described loosely:

- **Single-threaded.** A real engine runs a feed handler and a matcher on separate threads with a lock-free queue between them. There is no concurrency here at all, and the handoff is where much of the real engineering lives.
- **No C++.** Rust covers the systems ground, but C++ is what most of these job specs ask for.
- **No pre-trade risk layer.** No max order size, price collars, position limits, or self-trade prevention.
- **One symbol.** One book, no instrument registry.
- **Live capture tested against a fake feed, not the real exchange.** A local websocket server speaking Binance's wire format, with a scripted sequence gap. It found a real bug on the first run: a closed socket raised an unhandled exception and killed the recorder, despite the module claiming to handle disconnects. What it does not cover is the real feed's rate limits, 24-hour connection recycling, and genuinely out-of-order traffic.
- **The tail is still allocator-bound.** Both engines show a multi-millisecond worst case, which is the event log growing, not the matching logic. That is why p99.9 is the quoted figure and why the max is shown rather than hidden.

## No claim of profitability

The research half of this project makes no claim of a profitable strategy, and it is built to make weak results visible rather than hide them. Costs are charged on every trade, and a one basis point cost is enough to erase a half basis point edge. An automated strategy search is scored with a Deflated Sharpe Ratio, which asks whether the winner beats what pure luck would produce after that many attempts: search 400 variants of a coin flip and the best one looks brilliant.

When I ran it, the agent rejected its own best finding as a search artifact. That is the correct outcome and the one I would expect. The claim worth making is the rigor, not the returns.

## What I would do next

The lock-free queue between a feed thread and the matching thread. It is the gap that costs the most, it is where the memory-ordering and false-sharing problems I have not yet had to solve actually live, and the existing differential harness would validate it for free.
