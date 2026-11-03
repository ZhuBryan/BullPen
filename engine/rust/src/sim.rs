//! The default agent-based market from backend/market_sim.py, in Rust.
//!
//! The Python simulation spends about half its time in the order book and
//! half in Python itself: the agents deciding what to do, and the loop that
//! hands out fills. A faster book alone can therefore make it at most about
//! 1.8x faster (Amdahl's law). This module moves the whole thing, agents and
//! loop, next to the Rust book, so a run never crosses into Python until it
//! is finished.
//!
//! It is a port, not a redesign, and it must be BIT-FOR-BIT identical to the
//! Python version for the same seed: the same random draws in the same order,
//! the same float arithmetic in the same order, the same trades, the same
//! cash to the last bit. That is what lets the Python simulation stay the
//! oracle: engine/test_sim_rust.py runs both and compares everything, and the
//! tests at the bottom of this file pin values that CPython produced. A port
//! that is "close" would be untestable, because any small drift (one extra
//! random draw, one rounding done differently) changes which orders match,
//! and from there every later price is different.
//!
//! Two things make that possible:
//!   * PyRandom below is CPython's own random number generator (the Mersenne
//!     Twister, MT19937), seeded the way `random.Random(seed)` seeds it, with
//!     the exact recipes CPython uses for random(), randint, choice and shuffle.
//!   * Every float expression is written in the same order as the Python one,
//!     because float arithmetic is not associative: (a * b) * c and
//!     a * (b * c) can differ in the last bit.
//!
//! Only the DEFAULT market is ported (two market makers, a momentum trader, a
//! mean reverter and two noise traders, as default_agents() builds them).
//! Simulations with custom agents (the A-S maker, the forecast maker) still
//! run in Python.

use crate::{Book, Event, OrderId, Side};

const TICK: f64 = 0.01; // dollars per tick
pub const START_PRICE: i64 = 10_000; // $100.00 in ticks
const START_CASH: f64 = 100_000.0; // dollars

// ---- CPython's random module ---------------------------------------------

const N: usize = 624;
const M: usize = 397;

/// MT19937 exactly as CPython's `random.Random` implements it.
pub struct PyRandom {
    mt: [u32; N],
    mti: usize,
}

impl PyRandom {
    /// Same state as `random.Random(seed)` for an int seed: CPython takes
    /// abs(seed), splits it into 32-bit words (least significant first) and
    /// feeds them to init_by_array. Zero becomes the single word [0].
    pub fn new(seed: i64) -> PyRandom {
        let mut n: u64 = seed.unsigned_abs();
        let mut key: Vec<u32> = Vec::new();
        while n > 0 {
            key.push((n & 0xffff_ffff) as u32);
            n >>= 32;
        }
        if key.is_empty() {
            key.push(0);
        }
        let mut r = PyRandom { mt: [0u32; N], mti: N };
        r.init_by_array(&key);
        r
    }

    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..N {
            let prev: u32 = self.mt[i - 1];
            self.mt[i] = 1812433253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        self.mti = N;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19650218);
        let len: usize = key.len();
        let mut i: usize = 1;
        let mut j: usize = 0;
        let mut k: usize = if N > len { N } else { len };
        while k > 0 {
            let prev: u32 = self.mt[i - 1];
            let mixed: u32 = (prev ^ (prev >> 30)).wrapping_mul(1664525);
            self.mt[i] = (self.mt[i] ^ mixed)
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            if j >= len {
                j = 0;
            }
            k -= 1;
        }
        k = N - 1;
        while k > 0 {
            let prev: u32 = self.mt[i - 1];
            let mixed: u32 = (prev ^ (prev >> 30)).wrapping_mul(1566083941);
            self.mt[i] = (self.mt[i] ^ mixed).wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            k -= 1;
        }
        self.mt[0] = 0x8000_0000;
    }

    /// One raw 32-bit output (CPython's genrand_uint32).
    fn next_u32(&mut self) -> u32 {
        if self.mti >= N {
            // Regenerate all 624 words in place. Indices wrap with % N, which
            // reads the already-updated words exactly where the reference
            // implementation does.
            for kk in 0..N {
                let y: u32 = (self.mt[kk] & 0x8000_0000) | (self.mt[(kk + 1) % N] & 0x7fff_ffff);
                let mut v: u32 = self.mt[(kk + M) % N] ^ (y >> 1);
                if y & 1 != 0 {
                    v ^= 0x9908_b0df;
                }
                self.mt[kk] = v;
            }
            self.mti = 0;
        }
        let mut y: u32 = self.mt[self.mti];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// random.random(): a float in [0, 1) built from 53 random bits.
    pub fn random(&mut self) -> f64 {
        let a: u32 = self.next_u32() >> 5;
        let b: u32 = self.next_u32() >> 6;
        ((a as f64) * 67108864.0 + (b as f64)) * (1.0 / 9007199254740992.0)
    }

    /// random.getrandbits(k) for 0 <= k <= 32.
    pub fn getrandbits(&mut self, k: u32) -> u64 {
        if k == 0 {
            return 0;
        }
        (self.next_u32() >> (32 - k)) as u64
    }

    /// random._randbelow(n): uniform in [0, n), by rejection. Draws k =
    /// bit_length(n) bits and retries while the result is >= n.
    pub fn randbelow(&mut self, n: u64) -> u64 {
        let k: u32 = 64 - n.leading_zeros();
        let mut r: u64 = self.getrandbits(k);
        while r >= n {
            r = self.getrandbits(k);
        }
        r
    }

    /// random.randint(a, b), both ends included.
    pub fn randint(&mut self, a: i64, b: i64) -> i64 {
        a + self.randbelow((b - a + 1) as u64) as i64
    }

    /// random.shuffle(x), Fisher-Yates from the back, as CPython does it.
    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        let mut i: usize = x.len();
        while i > 1 {
            i -= 1;
            let j: usize = self.randbelow((i + 1) as u64) as usize;
            x.swap(i, j);
        }
    }
}

/// Python's `a // b` on ints rounds toward minus infinity; Rust's `/` rounds
/// toward zero. They differ only for negative results.
fn floor_div(a: i64, b: i64) -> i64 {
    let q: i64 = a / b;
    if (a % b != 0) && ((a < 0) != (b < 0)) {
        q - 1
    } else {
        q
    }
}

// ---- agents ----------------------------------------------------------------

/// The four strategies of the default market. See backend/market_sim.py for
/// the finance behind each one; the logic here follows it line by line.
enum Kind {
    /// Quotes both sides around the microprice of everyone else's orders and
    /// skews its quotes to shed inventory.
    Maker { spread: i64, size: i64, max_inv: i64 },
    /// Trend follower on an exponentially weighted average of mid changes.
    Momentum { lookback: i64, size: i64, trend: f64, seen: usize },
    /// Fades moves away from an exponentially weighted average price.
    Reverter { lookback: i64, size: i64, band: i64, avg: Option<f64>, seen: usize },
    /// Uninformed random orders.
    Noise { size: i64, spread: i64, rate: f64 },
}

struct Agent {
    kind: Kind,
    cash: f64,
    position: i64,
    live: Vec<OrderId>, // resting order ids (only makers use these)
}

impl Agent {
    fn new(kind: Kind) -> Agent {
        Agent { kind, cash: START_CASH, position: 0, live: Vec::new() }
    }

    /// The exchange tells us we traded. Same float steps as Agent.on_fill:
    /// value = price_ticks * TICK * qty, left to right.
    fn on_fill(&mut self, side: Side, price: i64, qty: i64) {
        let value: f64 = (price as f64) * TICK * (qty as f64);
        match side {
            Side::Buy => {
                self.cash -= value;
                self.position += qty;
            }
            Side::Sell => {
                self.cash += value;
                self.position -= qty;
            }
        }
    }
}

enum Order {
    Limit(Side, i64, i64), // side, price ticks, qty
    Market(Side, i64),     // side, qty
}

/// What one agent wants to do this step. `mid` is the mid measured at the
/// start of the step; `book` is the book as the agents before this one left it.
fn act(agent: &mut Agent, book: &Book, mid: i64, history: &[i64], rng: &mut PyRandom) -> Vec<Order> {
    let position: i64 = agent.position;
    let mut orders: Vec<Order> = Vec::new();
    match &mut agent.kind {
        Kind::Maker { spread, size, max_inv } => {
            let spread: i64 = *spread;
            let size: i64 = *size;
            let max_inv: i64 = *max_inv;
            let b: Vec<(i64, i64)> = book.depth(Side::Buy, 1);
            let a: Vec<(i64, i64)> = book.depth(Side::Sell, 1);
            // Fair value: the microprice of the top of book,
            // floor((bq * ap + aq * bp) / (bq + aq) + 0.5), or the mid if a
            // side is empty.
            let fv: i64 = if b.is_empty() || a.is_empty() {
                mid
            } else {
                let (bp, bq) = b[0];
                let (ap, aq) = a[0];
                let num: f64 = (bq * ap + aq * bp) as f64;
                let den: f64 = (bq + aq) as f64;
                (num / den + 0.5).floor() as i64
            };
            // int(spread * (position / max_inv)): `as i64` truncates toward
            // zero, like Python's int().
            let skew: i64 = if max_inv != 0 {
                ((spread as f64) * ((position as f64) / (max_inv as f64))) as i64
            } else {
                0
            };
            let mut bid: i64 = fv - spread - skew;
            let mut ask: i64 = fv + spread - skew;
            // Stay passive: never quote at or through someone else's quote.
            if !a.is_empty() {
                bid = bid.min(a[0].0 - 1);
            }
            if !b.is_empty() {
                ask = ask.max(b[0].0 + 1);
            }
            bid = bid.max(1);
            ask = ask.max(bid + 1);
            if position < max_inv {
                orders.push(Order::Limit(Side::Buy, bid, size));
            }
            if position > -max_inv {
                orders.push(Order::Limit(Side::Sell, ask, size));
            }
        }
        Kind::Momentum { lookback, size, trend, seen } => {
            if history.len() < *seen {
                *trend = 0.0;
                *seen = 0;
            }
            let a: f64 = 2.0 / ((*lookback + 1) as f64);
            let start: usize = if *seen > 1 { *seen } else { 1 };
            for i in start..history.len() {
                *trend = (1.0 - a) * *trend + a * ((history[i] - history[i - 1]) as f64);
            }
            *seen = history.len();
            if (history.len() as i64) <= *lookback {
                return orders;
            }
            if *trend > 0.0 && position < 100 {
                orders.push(Order::Market(Side::Buy, *size));
            } else if *trend < 0.0 && position > -100 {
                orders.push(Order::Market(Side::Sell, *size));
            }
        }
        Kind::Reverter { lookback, size, band, avg, seen } => {
            if history.len() < *seen {
                *avg = None;
                *seen = 0;
            }
            let a: f64 = 2.0 / ((*lookback + 1) as f64);
            for i in *seen..history.len() {
                let x: f64 = history[i] as f64;
                *avg = match *avg {
                    None => Some(x),
                    Some(prev) => Some((1.0 - a) * prev + a * x),
                };
            }
            *seen = history.len();
            if (history.len() as i64) < *lookback {
                return orders;
            }
            let avg_now: f64 = match *avg {
                Some(v) => v,
                None => return orders,
            };
            let m: f64 = mid as f64;
            let band_f: f64 = *band as f64;
            if m < avg_now - band_f && position < 100 {
                orders.push(Order::Market(Side::Buy, *size));
            } else if m > avg_now + band_f && position > -100 {
                orders.push(Order::Market(Side::Sell, *size));
            }
        }
        Kind::Noise { size, spread, rate } => {
            if rng.random() > *rate {
                return orders;
            }
            // choice((BUY, SELL)): index 0 is BUY.
            let side: Side = if rng.randbelow(2) == 0 { Side::Buy } else { Side::Sell };
            if rng.random() < 0.3 {
                orders.push(Order::Market(side, *size));
                return orders;
            }
            let off: i64 = rng.randint(1, *spread);
            let price: i64 = match side {
                Side::Buy => mid - off,
                Side::Sell => mid + off,
            };
            orders.push(Order::Limit(side, price, *size));
        }
    }
    orders
}

// ---- the simulation --------------------------------------------------------

/// One candle: first, highest, lowest and last trade price (ticks) in a step
/// that traded, and how many trades.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bar {
    pub open: i64,
    pub high: i64,
    pub low: i64,
    pub close: i64,
    pub count: usize,
}

pub struct Sim {
    book: Book,
    agents: Vec<Agent>,
    rng: PyRandom,
    /// Who owns each order: owner[id - 1] = (agent index, side). Ids are
    /// issued 1, 2, 3... so a Vec is enough; the Python dict needs pruning,
    /// this never grows faster than one entry per order.
    owner: Vec<(usize, Side)>,
    pub last: i64,
    pub history: Vec<i64>,
    pub bars: Vec<Bar>,
    seen_events: usize,
    pub trades: usize,
    start_cash: f64,
}

impl Sim {
    /// default_agents() in the same order, so agent i here is agent i there.
    pub fn new(seed: i64) -> Sim {
        let agents: Vec<Agent> = vec![
            Agent::new(Kind::Maker { spread: 4, size: 12, max_inv: 200 }), // Quill
            Agent::new(Kind::Maker { spread: 6, size: 8, max_inv: 200 }),  // Vane
            Agent::new(Kind::Momentum { lookback: 8, size: 6, trend: 0.0, seen: 0 }), // Chase
            Agent::new(Kind::Reverter { lookback: 25, size: 6, band: 8, avg: None, seen: 0 }), // Anchor
            Agent::new(Kind::Noise { size: 4, spread: 15, rate: 0.5 }), // Static
            Agent::new(Kind::Noise { size: 3, spread: 15, rate: 0.5 }), // Drift
        ];
        let mut start_cash: f64 = 0.0;
        for a in &agents {
            start_cash += a.cash;
        }
        Sim {
            book: Book::deterministic(),
            agents,
            rng: PyRandom::new(seed),
            owner: Vec::new(),
            last: START_PRICE,
            history: vec![START_PRICE],
            bars: Vec::new(),
            seen_events: 0,
            trades: 0,
            start_cash,
        }
    }

    fn mid_of(bb: Option<i64>, ba: Option<i64>, last: i64) -> i64 {
        match (bb, ba) {
            (Some(b), Some(a)) => floor_div(b + a, 2),
            _ => last,
        }
    }

    pub fn mid(&self) -> i64 {
        Sim::mid_of(self.book.best_bid(), self.book.best_ask(), self.last)
    }

    /// (position, cash) per agent, in default_agents() order.
    pub fn agents(&self) -> Vec<(i64, f64)> {
        self.agents.iter().map(|a| (a.position, a.cash)).collect()
    }

    pub fn step(&mut self) -> Result<(), String> {
        let mid: i64 = self.mid();

        // Makers cancel and requote every step, oldest order first.
        for idx in 0..self.agents.len() {
            if let Kind::Maker { .. } = self.agents[idx].kind {
                let ids: Vec<OrderId> = std::mem::take(&mut self.agents[idx].live);
                for id in ids {
                    self.book.cancel(id);
                }
            }
        }

        // Random turn order, so no agent gets a permanent time edge.
        let mut order: Vec<usize> = (0..self.agents.len()).collect();
        self.rng.shuffle(&mut order);
        for idx in order {
            let orders: Vec<Order> = act(
                &mut self.agents[idx],
                &self.book,
                mid,
                &self.history,
                &mut self.rng,
            );
            for o in orders {
                match o {
                    Order::Limit(side, price, qty) => {
                        let id: OrderId = self.book.limit(side, price, qty);
                        self.owner.push((idx, side));
                        debug_assert_eq!(id as usize, self.owner.len());
                        self.agents[idx].live.push(id);
                    }
                    Order::Market(side, qty) => {
                        let id: OrderId = self.book.market(side, qty);
                        self.owner.push((idx, side));
                        debug_assert_eq!(id as usize, self.owner.len());
                    }
                }
            }
        }

        // Settle: credit every new trade to both sides, aggressor first.
        let mut first: i64 = 0;
        let mut high: i64 = 0;
        let mut low: i64 = 0;
        let mut close: i64 = 0;
        let mut count: usize = 0;
        let end: usize = self.book.events.len();
        for i in self.seen_events..end {
            let trade: Option<(i64, i64, u64, u64)> = match &self.book.events[i] {
                Event::Trade { price, qty, aggressor, resting, .. } => {
                    Some((*price, *qty, *aggressor, *resting))
                }
                _ => None,
            };
            if let Some((price, qty, aggressor, resting)) = trade {
                let (a_idx, a_side) = self.owner[(aggressor - 1) as usize];
                self.agents[a_idx].on_fill(a_side, price, qty);
                let (r_idx, r_side) = self.owner[(resting - 1) as usize];
                self.agents[r_idx].on_fill(r_side, price, qty);
                if count == 0 {
                    first = price;
                    high = price;
                    low = price;
                }
                high = high.max(price);
                low = low.min(price);
                close = price;
                count += 1;
            }
        }
        self.seen_events = end;
        self.trades += count;
        if count > 0 {
            self.last = close;
            self.bars.push(Bar { open: first, high, low, close, count });
        }

        let bb: Option<i64> = self.book.best_bid();
        let ba: Option<i64> = self.book.best_ask();
        let new_mid: i64 = Sim::mid_of(bb, ba, self.last);
        self.history.push(new_mid);
        self.check(bb, ba)
    }

    /// The closed-economy invariants, same as Simulation._check: shares and
    /// money only move between agents, and the book is never crossed.
    fn check(&self, bb: Option<i64>, ba: Option<i64>) -> Result<(), String> {
        let mut pos: i64 = 0;
        let mut cash: f64 = 0.0;
        for a in &self.agents {
            pos += a.position;
            cash += a.cash;
        }
        if pos != 0 {
            return Err(format!("shares created from nothing: net position {}", pos));
        }
        if (cash - self.start_cash).abs() >= 1e-6 {
            return Err(format!("money created: {:.4} != {:.4}", cash, self.start_cash));
        }
        if let (Some(b), Some(a)) = (bb, ba) {
            if b >= a {
                return Err(format!("crossed book {}/{}", b, a));
            }
        }
        Ok(())
    }

    pub fn run(&mut self, steps: usize) -> Result<(), String> {
        for _ in 0..steps {
            self.step()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every expected value below was produced by CPython 3.10 (and checked
    // again on 3.14): random.Random(seed) for the RNG, and the Python
    // Simulation on the Python engine for the runs.

    #[test]
    fn rng_matches_cpython() {
        // (seed, random() bits x3, getrandbits(32) x2, randint(1,15) x5,
        //  choice((0,1)) x8, shuffle(range(8)))
        let cases: [(i64, [u64; 3], [u64; 2], [i64; 5], [u64; 8], [usize; 8]); 3] = [
            (
                0,
                [0x3feb0580f98a7dbe, 0x3fe84129978f9c1a, 0x3fdaeaa51052e978],
                [1112038970, 4146640122],
                [9, 8, 7, 15, 13],
                [1, 1, 1, 0, 0, 1, 0, 0],
                [2, 0, 3, 1, 6, 5, 7, 4],
            ),
            (
                7,
                [0x3fd4b9ad0f953a6e, 0x3fc34f0696513270, 0x3fe4d474883171ff],
                [311111475, 3527346212],
                [9, 2, 6, 10, 1],
                [0, 0, 0, 1, 1, 0, 0, 0],
                [3, 5, 2, 1, 7, 4, 0, 6],
            ),
            (
                1099511640121, // 2^40 + 12345: a seed wider than 32 bits
                [0x3fd3dcf982cb56aa, 0x3fbd69a2201fed28, 0x3fe1133e688e691c],
                [876331094, 593456706],
                [13, 14, 11, 2, 10],
                [1, 0, 0, 1, 1, 1, 0, 0],
                [5, 2, 1, 4, 0, 3, 7, 6],
            ),
        ];
        for (seed, floats, bits, ints, choices, shuffled) in cases.iter() {
            let mut r = PyRandom::new(*seed);
            for want in floats.iter() {
                assert_eq!(r.random().to_bits(), *want, "random(), seed {}", seed);
            }
            for want in bits.iter() {
                assert_eq!(r.getrandbits(32), *want, "getrandbits, seed {}", seed);
            }
            for want in ints.iter() {
                assert_eq!(r.randint(1, 15), *want, "randint, seed {}", seed);
            }
            for want in choices.iter() {
                assert_eq!(r.randbelow(2), *want, "choice, seed {}", seed);
            }
            let mut x: Vec<usize> = (0..8).collect();
            r.shuffle(&mut x);
            assert_eq!(&x[..], &shuffled[..], "shuffle, seed {}", seed);
        }
        // CPython seeds with abs(seed).
        assert_eq!(PyRandom::new(-7).random().to_bits(), PyRandom::new(7).random().to_bits());
    }

    #[test]
    fn floor_div_matches_python() {
        assert_eq!(floor_div(7, 2), 3);
        assert_eq!(floor_div(-7, 2), -4);
        assert_eq!(floor_div(-8, 2), -4);
        assert_eq!(floor_div(20_001, 2), 10_000);
    }

    fn check_run(seed: i64, steps: usize, trades: usize, last: i64, hist_last: i64,
                 hist_sum: i64, bars: usize, pos: [i64; 6], cash: [u64; 6]) {
        let mut s = Sim::new(seed);
        s.run(steps).expect("invariant broken");
        assert_eq!(s.trades, trades, "trades, seed {}", seed);
        assert_eq!(s.last, last, "last, seed {}", seed);
        assert_eq!(*s.history.last().unwrap(), hist_last, "final mid, seed {}", seed);
        assert_eq!(s.history.iter().sum::<i64>(), hist_sum, "mid path, seed {}", seed);
        assert_eq!(s.bars.len(), bars, "bars, seed {}", seed);
        let got = s.agents();
        for i in 0..6 {
            assert_eq!(got[i].0, pos[i], "agent {} position, seed {}", i, seed);
            assert_eq!(got[i].1.to_bits(), cash[i], "agent {} cash, seed {}", i, seed);
        }
    }

    #[test]
    fn simulation_matches_python_bit_for_bit() {
        check_run(1, 1, 0, 10_000, 10_000, 20_000, 0, [0; 6], [0x40f86a0000000000; 6]);
        check_run(
            7, 300, 546, 9977, 9993, 3_014_530, 267,
            [48, 2, -77, -16, 33, 10],
            [0x40f7404fd70a3d72, 0x40f85f611eb851eb, 0x40fa447f5c28f5c0,
             0x40f8cf08a3d70a41, 0x40f79c0e66666666, 0x40f82cb8a3d70a3c],
        );
        check_run(
            42, 2000, 3632, 9976, 9988, 19_894_334, 1698,
            [-53, 3, 100, -104, -71, 125],
            [0x40f9b9b63d70a3d8, 0x40f85f799999999a, 0x40f5d6cfae147ac9,
             0x40faf0bd70a3d707, 0x40fa2eac00000015, 0x40f56c970a3d70a9],
        );
    }
}
