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
  updates them together; here they are one map of `Level{head, tail, live}`,
  which halves the tree lookups and makes it impossible for the two to drift
  apart.
- **Orders**: a flat `std::vector<Order>` slab. Ids are issued 1, 2, 3... and
  each order takes the next slot, so order `id` is in slot `id - 1` and a
  cancel is an array index. No hash map.
- **Queues**: each level's FIFO is an intrusive singly linked list threaded
  through the slab (`Order::next`), so joining or leaving a queue never
  allocates. (Rust now does the same; Python still uses a deque per level. The event
  logs are identical either way, which is what `diff_test.py` checks.)
- **Preallocation**: `Book::reserve(orders, events)` sizes the slab and event
  log for a session and touches the memory up front, the way a venue sizes its
  memory at startup. The bench calls it; a book that outgrows it still works.
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

## Where the speed came from

The first version used a `std::deque` per level and an `std::unordered_map`
for cancels, and its p99 was 4.4 us against a 180 ns median. Timing every op
and sorting the slowest showed two causes, neither of them matching logic:

- The worst ops (up to 12 ms) were plain limit orders that happened to make
  the event log or slab double, copying everything written so far.
- The p99 itself was page faults. An `Event` is about 56 bytes, so a fresh
  4 KB page gets written once every ~70 events, roughly 1.4% of ops, and the
  first write to a page costs the kernel a few microseconds.

Replacing the deque and hash map with the intrusive list and slot = id - 1
gave about 20% more throughput and barely moved the tail. `reserve()` fixed
the tail: same 200,000-op file, same VM, median of 3 runs each, old and new
interleaved:

| | before | after | |
|---|---|---|---|
| throughput | 1.92M ops/sec | 3.37M ops/sec | 1.8x |
| p50 | 181 ns | 130 ns | |
| p90 | 621 ns | 341 ns | |
| p99 | 4,378 ns | 741 ns | 5.9x |
| p99.9 | 11,661 ns | 1,553 ns | 7.5x |
| max | 9.9 ms | 0.23 ms | |

Both emitted 239,703 events and `diff_test.py` still matches Python exactly.

## What I would change for more speed

Measured first, not assumed; these are the likely wins in order:

1. **Flat price arrays for a bounded tick range.** Most instruments trade in a
   narrow band of ticks. An array indexed by `price - base` with a tracked best
   index replaces the red-black tree's pointer chasing with one cache line, at
   the cost of a fallback for prices outside the band.
2. **A doubly linked queue.** With a `prev` index too, a cancel could unlink
   in O(1) and lazy husks would disappear. Not obviously a win: lazy cancel is
   already O(1), and husks cost only a skipped node.
3. **A ring for events.** The log keeps every event forever; a production
   engine writes them to a ring a separate thread drains to disk, so memory
   stays flat for a session of any length.
4. A smaller `Order`: `std::optional<Price>` costs 16 bytes where a sentinel
   price of 0 (already illegal) would do.

## Benchmark

Same 200,000-op file (`python diff_test.py gen big.txt 200000`), same machine,
same session. Both engines emitted 239,703 events, the check that they did the
same work. C++ figures are the median of 3 runs (ranges in brackets); Python
is `bench.py --repeats 3`, median of 3.

| | Python | C++ | ratio |
|---|---|---|---|
| throughput | 253,462 ops/sec | 3,372,458 ops/sec [3.06M-3.44M] | 13x |
| ns per op | 3,945 | 297 | 13x |
| p50 | 2,304 ns | 130 ns | 18x |
| p90 | 4,198 ns | 341 ns | 12x |
| p99 | 14,888 ns | 741 ns | 20x |
| p99.9 | 72,555 ns | 1,553 ns | 47x |
| max | 96 ms | 0.23 ms | |

Timer overhead was about 40 ns (steady_clock, 1 ns period) for C++ and 100 ns
for Python, so the C++ p50 is about 3x the timer floor: usable, but quote
throughput first. (Before the changes above this table read 1.8M ops/sec and
9x.)

**Caveat:** these were measured in a shared 2-vCPU Linux VM (g++ 11.4), where
other tenants add noise to every percentile. The Rust numbers in
`../README.md` (3.4M ops/sec) were measured on a different machine, so this
table does not say C++ is slower than Rust; rerun both on one machine before
comparing them.
