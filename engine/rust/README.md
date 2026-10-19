# Rust matching engine

A port of `../orderbook.py`. Same semantics, different data structures, verified by comparing event logs rather than by trusting that they match.

Standard library only by default. No dependencies to compile, and no part of the benchmark is attributable to someone else's code. The Python binding (pyo3) is an optional cargo feature, `python`, that only maturin turns on; criterion lives in a separate crate, `bench/`.

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

## Use it from Python (PyO3)

The same engine builds as a Python module, `bullpen_engine`, exposing the portable book interface from SPEC.md section 7.1. `engine/fastbook.py` then hands out the Rust `Book` automatically, falling back to the Python one when the module is not installed (or when `BULLPEN_ENGINE=python` is set).

From `engine/rust`:

```
python -m pip install .
python ../test_pyo3.py
python ../fastbook.py
```

`pip install .` reads `pyproject.toml`, fetches maturin into a temporary build environment, compiles with `--features python` in release mode, and installs the module. That sidesteps two common snags: a `maturin` command that isn't on PATH, and `maturin develop` refusing to run outside a virtualenv. Inside a virtualenv, `maturin develop --release` does the same thing faster on rebuilds. `test_pyo3.py` drives 20,000 random ops through both engines and compares every event tuple; it skips (exit 0) if the module is not built. The wheel uses Python's stable ABI (`abi3`), so one build works on CPython 3.9 and newer.

Do not run `cargo test --features python`: the `extension-module` setting deliberately leaves libpython unlinked, so a test binary cannot link. Plain `cargo test` is unaffected.

On Windows PowerShell, run the commands one per line (or join them with `;`). Do not use `&&`; Windows PowerShell 5 rejects it.

## Criterion benchmarks

```
cd bench
cargo bench
```

See `bench/README.md` for what each benchmark measures. Criterion gives a confidence interval for the mean and flags regressions between runs; `cargo run --release -- bench` above is still the tool for tail latency.

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

Both harnesses use the same methodology: warmup discarded, throughput measured as wall time over N operations so no per-op timer cost leaks into it, latency percentiles measured per operation with the timer's own median cost printed alongside, and a fixed seed. Quote p50, p99 and p99.9: in trading the tail is the number that matters, because the slow path is the one that runs when the market is moving.
