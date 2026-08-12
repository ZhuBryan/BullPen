# Rust matching engine

A port of `../orderbook.py`. Same semantics, different data structures, verified by comparing event logs rather than by trusting that they match.

Standard library only. No dependencies to resolve, and no part of the benchmark is attributable to someone else's code.

## Build and run

```bash
cargo test                      # correctness, including a randomized stream
cargo build --release
cargo run --release -- bench    # throughput and tail latency
```

## Prove it matches Python

The point of a rewrite is speed *without* a behaviour change, so the two engines are fed the same op file and their event logs are diffed. A rewrite that is fast but subtly different is worse than no rewrite, because everything built on it is quietly wrong.

```bash
cargo build --release
cd ..
python diff_test.py check ops.txt 20000
```

Both engines assign order ids sequentially, so identical input produces identical ids and the logs compare directly. Generating a stream separately in each language would not work: two RNGs never agree.

## What differs from the Python version, and why

| | Python | Rust |
|---|---|---|
| price levels | `dict` + a `bisect`-maintained sorted list | `BTreeMap` |
| orders | individual objects | one flat `Vec` slab, referenced by index |
| id lookup | `dict` | `HashMap` |
| cancellation | lazy, per-level live counter | identical |

The sorted-list trick exists in Python only because `max(dict)` there is an interpreted loop over every price level, which dominated the tail latency. `BTreeMap::last_key_value` is O(log n) and compiled, so Rust needs no such workaround. The slab keeps orders contiguous instead of scattering them across separately allocated objects, which is the layout change that actually matters for cache behaviour.

## Reporting the numbers honestly

Run the Python baseline (`python ../bench.py`) and this benchmark **on the same machine**, in the same session. A ratio between numbers measured on different hardware, or against a cloud sandbox with noisy neighbours, is not a result.

Both harnesses use the same methodology: warmup discarded, throughput measured as wall time over N operations so no per-op timer cost leaks into it, latency percentiles measured per operation with the timer's own median cost printed alongside, and a fixed seed. Quote p50, p99 and p99.9 — in trading the tail is the number that matters, because the slow path is the one that runs when the market is moving.
