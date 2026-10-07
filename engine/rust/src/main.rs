//! Benchmark and differential-test harness.
//!
//!   cargo run --release -- bench           throughput + tail latency
//!   cargo run --release -- replay ops.txt  emit the event log for diffing
//!
//! The benchmark deliberately mirrors bench.py: warmup discarded, throughput
//! measured as wall time over N ops (no per-op timer overhead), latency
//! percentiles measured per op with the timer's own cost reported alongside,
//! and a fixed seed so the order stream is identical across runs AND across
//! languages. Comparing two engines on different input proves nothing.

use bullpen_engine::{Book, Side};
use std::collections::VecDeque;
use std::env;
use std::fs;
use std::time::Instant;

/// Same LCG as the Python generator, so `bench` runs comparable input.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
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

/// Copy so the hot loops can `match *op` without moving out of a borrow.
#[derive(Clone, Copy)]
enum Op {
    Limit(Side, i64, i64),
    Market(Side, i64),
    Cancel,          // cancel whatever this run most recently rested
    CancelId(u64),   // cancel a specific id, used when replaying an op file
}

fn make_stream(n: usize, seed: u64) -> Vec<Op> {
    let mut rng = Lcg::new(seed);
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
    let mut live: Vec<u64> = Vec::new();
    for op in ops {
        apply(&mut book, &mut live, *op);
    }
    book
}

/// Parse the shared op-file format so both engines can be benchmarked on
/// byte-identical input. Generating a stream separately in each language makes
/// the workloads merely similar, which is not good enough to quote a ratio.
fn load_ops(path: &str) -> Vec<Op> {
    let text = fs::read_to_string(path).expect("could not read op file");
    let mut ops = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        let side = |s: &str| if s == "B" { Side::Buy } else { Side::Sell };
        match f[0] {
            "L" => ops.push(Op::Limit(side(f[1]), f[2].parse().unwrap(), f[3].parse().unwrap())),
            "M" => ops.push(Op::Market(side(f[1]), f[2].parse().unwrap())),
            "X" => ops.push(Op::CancelId(f[1].parse().unwrap())),
            other => panic!("unknown op {}", other),
        }
    }
    ops
}

fn timer_overhead(samples: usize) -> u128 {
    let mut v = Vec::with_capacity(samples);
    for _ in 0..samples {
        let t = Instant::now();
        let d = t.elapsed();
        v.push(d.as_nanos());
    }
    v.sort_unstable();
    v[v.len() / 2]
}

fn bench(ops_in: Option<Vec<Op>>, n: usize, warmup: usize) {
    println!("warmup {} ops (discarded)", warmup);
    let _ = run_stream(&make_stream(warmup, 1));

    let oh = timer_overhead(20_000);
    println!("timer overhead (median): {} ns", oh);
    if oh == 0 {
        println!("  NOTE: the clock cannot resolve two back-to-back calls here.");
        println!("  Windows QueryPerformanceCounter ticks at roughly 100ns, so any");
        println!("  per-op figure near that is at the measurement floor. The");
        println!("  throughput number below does not use a per-op timer and is");
        println!("  therefore the more trustworthy one.");
    }
    println!();

    let n = ops_in.as_ref().map(|v| v.len()).unwrap_or(n);
    let source = if ops_in.is_some() { "op file" } else { "generated" };
    println!("workload: {} ({} ops)\n", source, n);

    // --- throughput: no per-op timing, so no timer cost in the number ---
    let ops = ops_in.clone().unwrap_or_else(|| make_stream(n, 42));
    let t0 = Instant::now();
    let book = run_stream(&ops);
    let elapsed = t0.elapsed();
    let secs = elapsed.as_secs_f64();
    println!("THROUGHPUT  {} ops in {:.4}s", n, secs);
    println!(
        "            {:.0} ops/sec  ({:.0} ns/op)",
        n as f64 / secs,
        elapsed.as_nanos() as f64 / n as f64
    );
    println!("            {} events emitted\n", book.events.len());

    // --- latency percentiles: each op timed individually ---
    let mut book = Book::new();
    let mut live: Vec<u64> = Vec::new();
    let mut lat: Vec<u128> = Vec::with_capacity(n);
    for op in &ops {
        let t = Instant::now();
        match *op {
            Op::Limit(side, price, qty) => live.push(book.limit(side, price, qty)),
            Op::Market(side, qty) => {
                book.market(side, qty);
            }
            Op::Cancel => {
                if let Some(id) = live.pop() {
                    book.cancel(id);
                } else {
                    continue;
                }
            }
            Op::CancelId(id) => {
                book.cancel(id);
            }
        }
        lat.push(t.elapsed().as_nanos());
    }
    lat.sort_unstable();
    let pct = |p: f64| lat[((lat.len() as f64 * p) as usize).min(lat.len() - 1)];
    println!("LATENCY     per-op, timed individually (n={})", lat.len());
    for (name, p) in [("p50", 0.50), ("p90", 0.90), ("p99", 0.99), ("p99.9", 0.999)] {
        println!("            {:>5}: {:>9} ns", name, pct(p));
    }
    println!("            {:>5}: {:>9} ns\n", "max", lat[lat.len() - 1]);

    println!("note: single thread, release build. Compare against the Python");
    println!("      baseline in ../README.md on the SAME machine, or the ratio");
    println!("      is meaningless.");
}

/// Replay an op file and print the event log, for diffing against Python.
/// Format, one op per line:  `L B 9998 40` | `M S 30` | `X 42`
fn replay(path: &str) {
    let text = fs::read_to_string(path).expect("could not read op file");
    // Deterministic timestamps: this log is for comparison, so it must be
    // reproducible rather than reflecting whatever the clock said.
    let mut book = Book::deterministic();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        match f[0] {
            "L" => {
                let side = if f[1] == "B" { Side::Buy } else { Side::Sell };
                book.limit(side, f[2].parse().unwrap(), f[3].parse().unwrap());
            }
            "M" => {
                let side = if f[1] == "B" { Side::Buy } else { Side::Sell };
                book.market(side, f[2].parse().unwrap());
            }
            "X" => {
                book.cancel(f[1].parse().unwrap());
            }
            other => panic!("unknown op {}", other),
        }
        book.check();
    }
    let mut out = String::with_capacity(book.events.len() * 24);
    for e in &book.events {
        out.push_str(&e.encode());
        out.push('\n');
    }
    print!("{}", out);
}

/// Two threads: a feed handler pushing orders, a matcher draining them.
///
/// This is the architecture a real engine uses, and the reason to measure it is
/// that the handoff is not free. Comparing the lock-free queue against a
/// `Mutex<VecDeque>` on identical work is the honest way to show whether the
/// lock-free version earns its complexity, rather than asserting that it does.
fn pipeline(ops_path: &str) {
    use bullpen_engine::spsc;
    use std::sync::{Arc, Mutex};
    use std::thread;

    let ops = load_ops(ops_path);
    let n = ops.len();
    println!("pipeline: {} ops, feed thread -> matching thread\n", n);

    // ---- lock-free SPSC ------------------------------------------------
    let (mut tx, mut rx) = spsc::channel::<Op>(8192);
    let feed_ops = ops.clone();
    let t0 = Instant::now();
    let feeder = thread::spawn(move || {
        let mut full_hits = 0u64;
        for op in feed_ops {
            let mut v = op;
            loop {
                match tx.push(v) {
                    Ok(()) => break,
                    Err(back) => {
                        // Backpressure: the matcher is behind. Count it instead
                        // of dropping the message.
                        v = back;
                        full_hits += 1;
                        std::hint::spin_loop();
                    }
                }
            }
        }
        full_hits
    });

    let mut book = Book::new();
    let mut live: Vec<u64> = Vec::new();
    let mut done = 0usize;
    while done < n {
        match rx.pop() {
            Some(op) => {
                apply(&mut book, &mut live, op);
                done += 1;
            }
            None => std::hint::spin_loop(),
        }
    }
    let lockfree = t0.elapsed();
    let full_hits = feeder.join().unwrap();

    println!("lock-free SPSC   {:>10.4}s  {:>12.0} ops/sec  (backpressure hits: {})",
             lockfree.as_secs_f64(), n as f64 / lockfree.as_secs_f64(), full_hits);
    println!("                 {} events", book.events.len());

    // ---- mutex baseline, same work -------------------------------------
    let q: Arc<Mutex<VecDeque<Op>>> = Arc::new(Mutex::new(VecDeque::with_capacity(8192)));
    let feed_ops = ops.clone();
    let qp = Arc::clone(&q);
    let t1 = Instant::now();
    let feeder = thread::spawn(move || {
        for op in feed_ops {
            loop {
                let mut guard = qp.lock().unwrap();
                if guard.len() < 8192 {
                    guard.push_back(op);
                    break;
                }
                drop(guard);
                std::hint::spin_loop();
            }
        }
    });

    let mut book2 = Book::new();
    let mut live2: Vec<u64> = Vec::new();
    let mut done = 0usize;
    while done < n {
        let next = q.lock().unwrap().pop_front();
        match next {
            Some(op) => {
                apply(&mut book2, &mut live2, op);
                done += 1;
            }
            None => std::hint::spin_loop(),
        }
    }
    let mutexed = t1.elapsed();
    feeder.join().unwrap();

    println!("Mutex<VecDeque>  {:>10.4}s  {:>12.0} ops/sec",
             mutexed.as_secs_f64(), n as f64 / mutexed.as_secs_f64());
    println!("                 {} events", book2.events.len());

    assert_eq!(
        book.events.len(),
        book2.events.len(),
        "the two queues must feed the matcher identical work"
    );
    println!("\nspeedup {:.2}x  (both produced identical event counts)",
             mutexed.as_secs_f64() / lockfree.as_secs_f64());
    println!("note: single run, one machine. The queue is not the bottleneck at");
    println!("      these rates, so treat this as a floor on the difference.");
}

/// One op against the book. Shared so every mode applies ops identically.
fn apply(book: &mut Book, live: &mut Vec<u64>, op: Op) {
    match op {
        Op::Limit(side, price, qty) => live.push(book.limit(side, price, qty)),
        Op::Market(side, qty) => {
            book.market(side, qty);
        }
        Op::Cancel => {
            if let Some(id) = live.pop() {
                book.cancel(id);
            }
        }
        Op::CancelId(id) => {
            book.cancel(id);
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("bench");
    match mode {
        "bench" => {
            // `bench <n>` generates a stream; `bench <file>` replays a shared
            // op file so Python and Rust are measured on identical work.
            match args.get(2) {
                Some(a) => match a.parse::<usize>() {
                    Ok(n) => bench(None, n, 20_000),
                    Err(_) => bench(Some(load_ops(a)), 0, 20_000),
                },
                None => bench(None, 200_000, 20_000),
            }
        }
        "replay" => {
            let path = args.get(2).expect("usage: replay <ops file>");
            replay(path);
        }
        "pipeline" => {
            let path = args.get(2).expect("usage: pipeline <ops file>");
            pipeline(path);
        }
        other => eprintln!(
            "unknown mode {} (expected bench | replay | pipeline)",
            other
        ),
    }
}
