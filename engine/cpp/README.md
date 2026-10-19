# Bullpen engine, C++17 port

The third implementation of the matching engine, after the Python oracle
(`../orderbook.py`) and the Rust port (`../rust/`). Same semantics, same event
log: `../diff_test.py` replays one op file through every engine and requires
the encoded logs to be identical line for line. Standard library only.

- `orderbook.hpp`: the engine (header-only)
- `main.cpp`: `replay`, `bench`, `selftest`
- `Makefile`

## Build

```bash
make          # bullpen_cpp: -O3 -DNDEBUG -std=c++17 -Wall -Wextra -Wpedantic -Werror
make test     # build, then run the selftest
make debug    # bullpen_cpp_debug: -O0 -g, AddressSanitizer + UBSan
make clean
```

The engine is one header and one translation unit, so no build system is
required:

```bat
:: MSVC, from a Developer Command Prompt
cl /std:c++17 /O2 /EHsc /W4 /Fe:bullpen_cpp.exe main.cpp

:: MinGW-w64
g++ -std=c++17 -O3 -DNDEBUG -Wall -Wextra -Wpedantic -Werror -o bullpen_cpp.exe main.cpp
```

No `-march=native`: the binary should run on any x86-64 machine it is copied to.

## Run

```bash
./bullpen_cpp selftest                  # 16 behavioural tests, exit 1 on failure
./bullpen_cpp replay ../ops.txt         # encoded event log, one line per event
./bullpen_cpp bench                     # 200,000 generated ops (same LCG as Rust)
./bullpen_cpp bench 50000               # N generated ops
./bullpen_cpp bench ops.txt             # an op file shared with the other engines

cd .. && python diff_test.py check ops.txt 20000
```

`check` compares Python against each compiled engine that is built (Rust and
C++), reports each separately, and fails if neither is built or if any built
one differs. Benchmark against Python on the same file:
`python bench.py --ops-file <file> --repeats 3`.

`bench` follows the same methodology as `bench.py` and the Rust bench: a
discarded warmup, throughput as wall time over N ops (no per-op timer in that
number), per-op latency percentiles from `std::chrono::steady_clock`, the
timer's own median overhead printed alongside, and a warning when that overhead
reads near zero (the clock cannot resolve the work, so trust throughput).

## Design

The data structures mirror the Rust port so that a speed comparison is about
the language and not the algorithm:

- **Price levels**: `std::map<Price, Level>` per side. Best bid is the last
  key, best ask the first. One deviation from Rust, behaviour-neutral: Rust
  keeps two parallel maps (price to queue, price to live count) and always
  updates them together; here they are one map of `Level{fifo, live}`, which
  halves the tree lookups and makes it impossible for the two to drift apart.
- **Orders**: a flat `std::vector<Order>` slab referenced by index, plus an
  `std::unordered_map<OrderId, slot>` for O(1) cancel.
- **Lazy cancellation**: a cancel flags the order and decrements the level's
  live count; matching discards the flagged husk when it reaches it. A level
  whose live count hits zero is erased at once, so it never advertises a
  price nobody can trade at.
- **Events**: a `std::variant` of the four Rust event types (accepted, trade,
  canceled, reject_cancel), each carrying `seq` and `ts_ns`. `std::visit` with
  `if constexpr` and a `static_assert` makes a forgotten event type a compile
  error, like a Rust `match`. Encoding uses `std::to_chars`: locale-free and
  allocation-free.
- **Clock**: `Book::Clock::Steady` (nanoseconds since the book was created) or
  `Book::Clock::Deterministic` (ts equals seq). `replay` uses deterministic, so
  the log is reproducible; `ts` is never part of the encoded line anyway.

Input rules match the engines. A non-positive price, non-positive qty or bad
side throws `std::invalid_argument` before any state changes, so a rejected
order leaves no event and consumes no id. Python also refuses a float even when
it is whole (`100.0`); in C++ a `double` argument would silently truncate, so
deleted template overloads of `limit`/`market` make it a compile error. The
op-file parser uses `std::from_chars` on whole tokens, so `100.5`, `100.0`,
`1e3`, `+5` and out-of-range values are rejected with a file and line number.
It is stricter than the Python and Rust readers in two ways that never arise
in generated files: a side other than `B`/`S` is an error (they treat it as a
sell) and extra fields are an error (they ignore them).

`cancel` is `[[nodiscard]]`: ignoring a failed cancel is a real trading bug
(the order may have filled), so callers that only want the log say so with a
`void` cast. `check_invariants()` throws rather than asserting, so it still
runs in the `-DNDEBUG` build. `replay` runs the full check after each of the
first 20,000 ops, as Rust does after every op, then every 1,000 ops and after
the last one; the full check is O(book), and per-op it makes a 200,000-op
replay quadratic. The O(1) checks (book not crossed, seq dense) run after every
op throughout.

## What I would change for more speed

Measured first, not assumed; these are the likely wins in order:

1. **Flat price arrays for a bounded tick range.** Most instruments trade in a
   narrow band of ticks. An array indexed by `price - base` with a tracked best
   index replaces the red-black tree's pointer chasing with one cache line, at
   the cost of a fallback for prices outside the band.
2. **Intrusive FIFO lists.** Store `prev`/`next` slot indices in the order
   itself instead of a `std::deque<std::size_t>` per level. Cancels could then
   unlink in O(1) and lazy husks would disappear, and a level would no longer
   own a separately allocated deque.
3. **Pool allocation and pre-sizing.** The event log and slab grow by doubling;
   the multi-millisecond `max` below is consistent with one of those
   reallocations copying several MB. Chunked event storage (or writing events
   to a ring and draining them) removes that tail; an open-addressing hash map
   with ids as near-dense keys (or simply a vector indexed by id) beats
   `std::unordered_map` node allocation.
4. A smaller `Order`: `std::optional<Price>` costs 16 bytes where a sentinel
   price of 0 (already illegal) would do.

## Benchmark

Same 200,000-op file (`python diff_test.py gen big.txt 200000`), same machine,
same session. Both engines emitted 239,703 events, the check that they did the
same work. C++ figures are the median of 3 runs (ranges in brackets); Python
is `bench.py --repeats 3`, median of 3.

| | Python | C++ | ratio |
|---|---|---|---|
| throughput | 200,739 ops/sec | 1,797,470 ops/sec [1.73M-2.00M] | 9.0x |
| ns per op | 4,982 | 556 | 9.0x |
| p50 | 3,277 ns | 180 ns | 18x |
| p90 | 5,771 ns | 701 ns | 8.2x |
| p99 | 23,854 ns | 4,318 ns | 5.5x |
| p99.9 | 93,445 ns | 13,886 ns | 6.7x |
| max | 35.9 ms | 7.8 ms | 4.6x |

Timer overhead was 40 ns (steady_clock, 1 ns period) for C++ and 90 ns for
Python, so the C++ p50 is about 4.5x the timer floor: usable, but quote
throughput first.

**Caveat:** these were measured in a shared 2-vCPU Linux VM (g++ 11.4), where
other tenants add noise to every percentile. The Rust numbers in
`../README.md` (3.4M ops/sec) were measured on a different machine, so this
table does not say C++ is slower than Rust; rerun both on one machine before
comparing them.
