# Criterion benchmarks

A separate crate on purpose: criterion and its dependencies never enter the engine's own build, so `cargo test` in `engine/rust` stays std-only.

```
cd engine/rust/bench
cargo bench
```

Run one group with a filter, e.g. `cargo bench -- market_sweep`. Run it twice: the second run reports whether each number changed and whether that change is statistically significant. Results are kept in `target/criterion/`.

| Bench | Measures |
|---|---|
| `mixed_stream/10k_ops` | Throughput (ops/s) over 10,000 mixed ops: 75% limits, 10% markets, 15% cancels. Same generator as `cargo run --release -- bench`. |
| `limit_insert/existing_level` | One non-crossing limit joining an existing price level (a queue push). |
| `limit_insert/new_level` | One non-crossing limit opening a new price level (a queue push plus a `BTreeMap` insert). |
| `cancel/mid_queue` | Cancelling an order in the middle of its queue. Lazy cancel, so a hash lookup, not a scan. |
| `market_sweep/{1,10,50}` | A market order that empties 1, 10 or 50 ask levels (5 orders each). Reported in fills/s, so a flat number means cost per fill does not grow with depth. |

The single-op benches build a fresh pre-filled book outside the timed region for every iteration, so they measure the operation, not a book that keeps growing.

Criterion reports a confidence interval for the **mean**. For tail latency (p99, p99.9) use `cargo run --release -- bench` in `engine/rust`, which times each op individually.
