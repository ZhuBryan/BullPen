//! Criterion benchmarks for the matching engine. See ../README.md.
//!
//! Criterion repeats each measurement many times and reports a confidence
//! interval for the mean, plus whether the result moved since the last run.
//! That is what you want for "did my change make it faster?". It does NOT
//! report per-op tail latency (p99, p99.9); `cargo run --release -- bench` in
//! the engine crate does that.
//!
//! `black_box` stops the optimiser from deleting work whose result is unused,
//! or from constant-folding an input it can see at compile time.
//!
//! The single-op benchmarks use `iter_batched`: a fresh, pre-filled book is
//! built for every iteration OUTSIDE the timed region, and returned from the
//! routine so its deallocation is not timed either. Otherwise every insert
//! would grow one ever-larger book and the benchmark would measure the book's
//! size, not the operation.

use std::hint::black_box;

use bullpen_engine::{Book, OrderId, Side};
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};

/// Same LCG and op mix as src/main.rs, so the numbers are comparable.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % ((hi - lo + 1) as u64)) as i64
    }
}

#[derive(Clone, Copy)]
enum Op {
    Limit(Side, i64, i64),
    Market(Side, i64),
    Cancel, // cancel the most recently rested order
}

/// 75% limits around a 10_000 mid, 10% market orders, 15% cancels.
fn make_stream(n: usize, seed: u64) -> Vec<Op> {
    let mut rng = Lcg(seed);
    let mut ops = Vec::with_capacity(n);
    for _ in 0..n {
        let side = if rng.next() % 2 == 0 { Side::Buy } else { Side::Sell };
        let roll = rng.range(0, 99);
        if roll < 15 {
            ops.push(Op::Cancel);
        } else if roll < 25 {
            ops.push(Op::Market(side, rng.range(1, 100)));
        } else {
            let off = rng.range(1, 12);
            let price = match side {
                Side::Buy => 10_000 - off,
                Side::Sell => 10_000 + off,
            };
            ops.push(Op::Limit(side, price, rng.range(1, 100)));
        }
    }
    ops
}

fn run_stream(ops: &[Op]) -> Book {
    let mut book = Book::new();
    let mut live: Vec<OrderId> = Vec::new();
    for op in ops {
        match *op {
            Op::Limit(side, price, qty) => live.push(book.limit(side, price, qty)),
            Op::Market(side, qty) => {
                book.market(side, qty);
            }
            Op::Cancel => {
                if let Some(id) = live.pop() {
                    book.cancel(id);
                }
            }
        }
    }
    book
}

/// A resting book that does not cross: `levels` bid prices below 10_000 and
/// `levels` ask prices above it, `per_level` orders of 10 shares at each.
/// Returns the book and the bid ids in arrival order.
fn seeded(levels: i64, per_level: usize) -> (Book, Vec<OrderId>) {
    let mut book = Book::new();
    let mut bids = Vec::new();
    for i in 0..levels {
        for _ in 0..per_level {
            bids.push(book.limit(Side::Buy, 9_999 - i, 10));
            book.limit(Side::Sell, 10_001 + i, 10);
        }
    }
    (book, bids)
}

/// (a) Throughput over a realistic mixed stream: matching, resting, cancels,
/// level creation and removal all interleaved. Reported as ops/second.
fn mixed_stream(c: &mut Criterion) {
    const N: usize = 10_000;
    let ops = make_stream(N, 42);
    let mut g = c.benchmark_group("mixed_stream");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function("10k_ops", |b| b.iter(|| run_stream(black_box(&ops))));
    g.finish();
}

/// (b) One non-crossing limit order into a 10x10-per-side book. Two cases,
/// because they take different paths: joining an existing price level is a
/// VecDeque push; opening a new one is also a BTreeMap insert.
fn limit_insert(c: &mut Criterion) {
    let mut g = c.benchmark_group("limit_insert");
    g.bench_function("existing_level", |b| {
        b.iter_batched(
            || seeded(10, 10).0,
            |mut book| {
                black_box(book.limit(Side::Buy, black_box(9_995), black_box(10)));
                book
            },
            BatchSize::SmallInput,
        )
    });
    g.bench_function("new_level", |b| {
        b.iter_batched(
            || seeded(10, 10).0,
            |mut book| {
                black_box(book.limit(Side::Buy, black_box(9_950), black_box(10)));
                book
            },
            BatchSize::SmallInput,
        )
    });
    g.finish();
}

/// (c) Cancel one order in the middle of a queue. Lazy cancellation makes this
/// a hash lookup plus a counter decrement, not a scan of the queue.
fn cancel(c: &mut Criterion) {
    // `seeded` is deterministic, so the ids are the same in every fresh book.
    let ids = seeded(10, 10).1;
    let target = ids[ids.len() / 2];
    c.bench_function("cancel/mid_queue", |b| {
        b.iter_batched(
            || seeded(10, 10).0,
            |mut book| {
                black_box(book.cancel(black_box(target)));
                book
            },
            BatchSize::SmallInput,
        )
    });
}

/// (d) A market buy that sweeps the whole ask side: `levels` price levels of 5
/// orders each, so levels*5 fills. Throughput is reported in fills/second, which
/// shows whether the cost per fill stays flat as the sweep gets deeper.
fn market_sweep(c: &mut Criterion) {
    const PER_LEVEL: usize = 5;
    let mut g = c.benchmark_group("market_sweep");
    for &levels in &[1i64, 10, 50] {
        let fills = levels as u64 * PER_LEVEL as u64;
        let qty = levels * PER_LEVEL as i64 * 10; // every ask is 10 shares
        g.throughput(Throughput::Elements(fills));
        g.bench_with_input(BenchmarkId::from_parameter(levels), &levels, |b, &levels| {
            b.iter_batched(
                || seeded(levels, PER_LEVEL).0,
                |mut book| {
                    black_box(book.market(Side::Buy, black_box(qty)));
                    book
                },
                BatchSize::SmallInput,
            )
        });
    }
    g.finish();
}

criterion_group!(benches, mixed_stream, limit_insert, cancel, market_sweep);
criterion_main!(benches);
