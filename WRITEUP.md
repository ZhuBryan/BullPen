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

## The handoff between threads

A single-threaded engine is a toy in one specific way: real venues parse the socket on one thread and match on another, and the queue between them is where the interesting failure modes live. A `Mutex<VecDeque>` works and is two lines, but it puts a kernel-visible lock on the hot path, and the producer can block the consumer for far longer than the work itself takes.

So the queue is a lock-free SPSC ring buffer. Three things make it correct, and each is something I have since been asked about:

**Ownership.** `head` (next write slot) is written only by the producer, `tail` only by the consumer. Each side loads its own index with `Relaxed`, because nobody else writes it, and the other side's with `Acquire`. The single-producer constraint is enforced by the type system rather than by a comment: `channel()` returns two separate owned handles, so a second producer is not constructible.

**Memory ordering.** The producer writes the slot, then publishes the new index with `Release`. The consumer reads the index with `Acquire`. That pairing is the entire correctness argument: it guarantees the slot write is visible to anyone who observes the index advertising it. Use `Relaxed` instead and the consumer may legally read uninitialized memory. The reason this bug is dangerous is that it will pass every test you write on x86, because x86 is strongly ordered enough to hide it, and then fail on ARM.

**False sharing.** `head` and `tail` sit on separate 64-byte cache lines. Without the padding they share one, so every producer write invalidates the consumer's cached copy of the line it is reading. Two threads touching no common data would still ping-pong a cache line between cores: logically correct, needlessly slow.

Backpressure returns the value to the caller rather than blocking or dropping it. A market-data feed that silently discards messages is worse than one that tells you it is behind.

The pipeline mode runs the same op file through the lock-free queue and through a mutex baseline, asserts both fed the matcher identical event counts, and prints the ratio. Benchmarking a lock-free structure against nothing proves it runs; benchmarking it against the obvious alternative on identical work shows whether the complexity earned anything.

It earned 1.19x: 2.50M ops/sec against 2.10M, with identical event counts. That's less than I expected, and the backpressure counter told me why before I had a chance to invent a story. The producer found the queue full 132,821 times while pushing 20,000 ops. The feed thread was mostly waiting on the matcher, so I was measuring a queue that wasn't the constraint. This is the same lesson as the cancel profiling, from the other side: measure where the time goes before deciding what to optimize. A faster handoff into a matcher that can't keep up mostly buys a faster wait.

## Tests that passed for the wrong reason

The second round of features (order-book backtests, a portfolio layer, a market maker, stylized-facts tests, a FIX session layer) all shipped with self-checks that passed. Then a reviewer who hadn't written any of it went looking for inputs the authors hadn't imagined, and four of those passing tests turned out to be testing the easy case.

**Risk parity.** The goal is weights where every asset contributes the same share of portfolio risk. The code used a popular fixed-point shortcut, and the test checked it on positively correlated assets, where it works. Give it a stock index next to its inverse ETF and it put zero weight on the index, with a 70/30 risk split. On 200 random covariance matrices with mixed-sign correlations it missed on 170. The fix was to solve the problem properly: minimize ½w'Cw minus the average log weight, whose optimum is exactly equal risk contributions, by coordinate descent. It's convex, so it converges for any valid covariance matrix, and the test now includes the inverse-ETF case and the 200 random matrices.

**The cost breakdown.** An order-book backtest trades at the next day's open, through a book, in whole shares. The idealized backtest trades at the close, frictionlessly, in fractional shares. I split the difference into "timing", "cost" and "capacity", and the three summed exactly, which felt like proof. But "timing" compared a whole-share run against a fractional one, so on a 20-year run of a stock that rose 20x, a $1,000 account's inability to buy fractions showed up as about 200 points of "timing". Summing exactly proves the bookkeeping, not the labels. Each term now compares two runs that differ in exactly one thing, rounding is its own term, and the test asserts the property that gives the game away: timing can't depend on account size.

**Autocorrelation.** One stylized fact of real markets is that returns aren't linearly predictable, which you check by comparing autocorrelations against a 95% band. The textbook band assumes independent returns. Real returns are uncorrelated but cluster in volatility, and under clustering that band is too narrow: a series built to be truly unpredictable "failed" 41 times in 100. Swapping in a band that allows for changing volatility brought that to 6, but testing the other direction found a second problem. Lag 1 had been excused entirely (to allow for bid-ask bounce), which hid exactly the lag where simple predictability lives, so a genuinely predictable series was caught only 23 times in 50. Bounce only ever produces *negative* lag-1 correlation, so now only that is excused, and detection went to 45 in 50.

**FIX ownership.** The venue recorded which client owned an order before checking whether the order was a duplicate. So a second client could submit a copy of someone else's order ID, get rejected, and still become its owner, then cancel the original and receive its reports. The happy-path test (one client, valid orders) could never see it. Neither could it see that a resend request starting at minus three million froze every session on the shared event loop.

The common thread is that a self-check written alongside the code tests what the author was thinking about. The fixes were each small. Finding them took someone asking "what's the nastiest input that's still legal?", and every one of those inputs is now a test.

## My simulated market was trading against itself

The simulator failed one of the most basic tests a market model can take: its returns were predictable. Real returns have almost no lag-1 autocorrelation; the simulator's was +0.35, meaning a rise this step strongly predicted a rise next step.

The obvious suspect was the momentum traders, and removing them did fix the number. But real markets have momentum traders too, and they still don't trend step to step, because liquidity providers absorb predictable flow. So deleting them would have been tuning the model to pass the test. Running each agent type out in turn, and checking the measurement itself, pointed somewhere else. The naive market maker centred its quotes on the current mid, and the current mid was mostly *its own previous quotes*. Its small inventory skew was applied again on top of the last one every step, so a price offset became a price velocity, and every move got locked in as the new reference for the momentum traders to chase. Centring the makers on the microprice of everyone else's orders took the correlation to about zero and the test to passing on 87% of held-out seeds, with heavy tails and volatility clustering intact.

Fixing that exposed a second bug in the Avellaneda-Stoikov maker. Its inventory term was counted in shares, while the model counts inventory in lots (each fill moves it by one). With 10-share orders the skew was ten times too strong: quotes sat on their safety clamps, and the maker pushed the price against its own position, which showed up as -8 ticks of adverse selection five steps after its fills. Counted in lots, those markouts turn positive and the maker runs near break-even, holding about a third of the naive maker's inventory. The model had been right all along; the units weren't.

## An AUC of 0.90 that's worth nothing

The first real data came from Kraken's BTC/USD order book: twelve minutes at first, then a full hour of about 65,000 book updates, every one verified against the exchange's checksum. On the first sample, the order-flow forecast predicted the direction of the mid price ten events ahead with an AUC of 0.90 and an information coefficient of 0.33, out of sample, t-stat 18.

A number like that is a reason to look for a bug, so I tested the timing both ways. If the features were peeking at the future, giving them one more event of future information should help; it added nothing (0.897). If the signal were a last-instant tell, making the features stale should kill it; twenty events of delay only took it to 0.865. So it's real, and it's a slow-moving state of the book.

Then the baseline that actually mattered. Not a coin flip, but the dumbest rule using the same information: "the last move continues". It tied the model on IC, 0.327 to 0.329. Over the full hour the model pulled modestly ahead (0.33 to 0.27, AUC 0.92), so it does learn something beyond this rule, but the shape of the answer didn't change. On this book, 82% of mid moves continue the previous one, because the mid moves in half-ticks: one side of the book requotes, then the other follows. Most of the "forecast" is the second half of a step that has already started. It's worth about 0.03 basis points per trade, against taker fees around 26.

So the honest result is a highly accurate forecast with no economic value, and the lesson is about baselines. Beating a coin flip proves nothing when the market's own mechanics are predictable; the pipeline now prints the no-model rule beside every model.

## Twelve times faster, twice as fast

The Rust engine matches orders 12.7x faster than the Python one, and the market simulation built on it got about 2x faster. Rather than guess why, I measured it two ways. cProfile is easy but charges a fixed cost to every Python call, which distorts exactly this kind of question. So the benchmark also records every call the simulation makes on the book, replays them in a bare loop, and times the Python side separately against a stand-in that answers each call from the recording and does no work. The two halves add up to the whole: the book is about 48% of a run.

That one number settles it. By Amdahl's law, an engine 12.7x faster at 48% of the work gives at most about 1.8x overall, and an infinitely fast one about 1.9x. Batching orders into one call per agent trims the cost of crossing from Python into Rust, but it can't move that ceiling. Only moving the agents themselves out of Python can. It's the cancel story again at a larger scale: measure where the time goes before deciding what to speed up.

## Honest limitations

The things this does not do, since a half-built project described accurately is worth more than a finished one described loosely:

- **The agents are still Python.** The engine exists in three languages, but the simulated traders don't, which is why the simulation is only ~2x faster on Rust (see above).
- **The matching engine is still single-threaded.** There is now a lock-free SPSC queue and a two-thread pipeline mode measured against a mutex baseline, but the deterministic single-threaded path remains what the differential test runs against, on purpose: an oracle has to be reproducible.
- **Multi-symbol, but not multi-threaded across symbols.** There is an instrument registry and a venue that routes each symbol to its own book, but one thread serves all of them. Sharding symbols across threads is the obvious scaling step, since books don't interact.
- **Live capture tested against a fake feed, not the real exchange.** A local websocket server speaking Binance's wire format, with a scripted sequence gap. It found a real bug on the first run: a closed socket raised an unhandled exception and killed the recorder, despite the module claiming to handle disconnects. What it does not cover is the real feed's rate limits, 24-hour connection recycling, and genuinely out-of-order traffic.
- **The tail is still allocator-bound.** Both engines show a multi-millisecond worst case, which is the event log growing, not the matching logic. That is why p99.9 is the quoted figure and why the max is shown rather than hidden.

## No claim of profitability

The research half of this project makes no claim of a profitable strategy, and it is built to make weak results visible rather than hide them. Costs are charged on every trade, and a one basis point cost is enough to erase a half basis point edge. An automated strategy search is scored with a Deflated Sharpe Ratio, which asks whether the winner beats what pure luck would produce after that many attempts: search 400 variants of a coin flip and the best one looks brilliant.

When I ran it, the agent rejected its own best finding as a search artifact. That is the correct outcome and the one I would expect. The claim worth making is the rigor, not the returns.

## What I would do next

Move the simulator's agents into compiled code, since the measurement above says that's the only way past a ~1.9x end-to-end gain, and do it with the same discipline as the engine ports: a differential test that holds the Python agents as the oracle.

After that, longer real recordings: a full day of BTC/USD across quiet and busy hours, the stylized facts run on real book returns next to the simulator's, and a forecast target that's actually tradeable (moves larger than the spread, net of fees) instead of the next tick.
