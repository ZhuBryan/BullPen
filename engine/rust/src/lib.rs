//! Limit order book matching engine, price-time priority.
//!
//! This is a port of `engine/orderbook.py`, and it is deliberately a port
//! rather than a redesign: the two implementations are fed identical order
//! streams and their event logs are compared byte for byte (see `--diff` in
//! main.rs). A rewrite that is fast but subtly different in behaviour is worth
//! nothing, so the Python version acts as the oracle and Rust has to match it.
//!
//! Where the two differ is only in data structures, and each difference is
//! there because the language made a different choice cheap:
//!
//! * PRICE LEVELS. Python keeps a `bisect`-maintained sorted list beside the
//!   dict, because `max(dict)` there is an interpreted loop over every level.
//!   Rust uses a `BTreeMap`, whose `first_key_value` / `last_key_value` are
//!   O(log n) and compiled. No workaround needed.
//!
//! * ORDER STORAGE. Orders live in one flat `Vec` (a slab) and are referenced
//!   by index, so they stay contiguous in memory instead of being scattered
//!   across individually allocated objects. Ids are issued 1, 2, 3... and
//!   each order takes the next slot, so order `id` is in slot `id - 1`: a
//!   cancel is an array index, with no hash map.
//!
//! * QUEUES. Each price level's FIFO is a singly linked list threaded through
//!   the slab (`Order::next`), with head and tail stored in the level. Joining
//!   or leaving a queue never allocates; a `VecDeque` per level allocated
//!   every time a new price level opened.
//!
//! * CANCELLATION. Same lazy scheme as Python: cancelling flags the order and
//!   decrements a per-level live counter, and matching skips the husks. The
//!   counter is what stops a level of pure husks from advertising a price.
//!
//! Prices are integer ticks. Never floats: 0.1 + 0.2 != 0.3 and money bugs from
//! float rounding are silent.

use std::collections::BTreeMap;
use std::time::Instant;

pub mod sim;
pub mod spsc;

// Python bindings (PyO3). Only compiled with `--features python`, so the
// default build and `cargo test` stay std-only. See src/py.rs.
#[cfg(feature = "python")]
pub mod py;

pub type Price = i64;
pub type Qty = i64;
pub type OrderId = u64;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Side {
    Buy,
    Sell,
}

impl Side {
    pub fn as_char(self) -> char {
        match self {
            Side::Buy => 'B',
            Side::Sell => 'S',
        }
    }
}

#[derive(Clone, Debug)]
pub struct Order {
    pub id: OrderId,
    pub side: Side,
    pub price: Option<Price>, // None = market order
    pub qty: Qty,
    pub remaining: Qty,
    pub cancelled: bool,
    /// Live on the book: what `has_order` reports.
    pub resting: bool,
    /// Next order in the same price level's queue (NO_SLOT at the back).
    next: usize,
}

const NO_SLOT: usize = usize::MAX;

/// Everything the engine emits. Formatted identically to the Python side so the
/// two logs can be diffed directly.
///
/// Every event carries a monotonic `seq` and a `ts_ns` timestamp, the way a real
/// venue's feed does. `seq` is the authoritative ordering because it is
/// deterministic and survives replay; `ts_ns` is nanoseconds since the book was
/// created and is wall-clock-ish, so it is deliberately NOT part of `encode()`
/// (see that method).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    Accepted {
        seq: u64,
        ts_ns: u64,
        id: OrderId,
        side: Side,
        price: Option<Price>,
        qty: Qty,
    },
    Trade {
        seq: u64,
        ts_ns: u64,
        price: Price,
        qty: Qty,
        aggressor: OrderId,
        resting: OrderId,
    },
    Canceled {
        seq: u64,
        ts_ns: u64,
        id: OrderId,
        remaining: Qty,
    },
    RejectCancel {
        seq: u64,
        ts_ns: u64,
        id: OrderId,
    },
}

impl Event {
    /// Wire form used for the differential test against the Python engine.
    ///
    /// `seq` is included because it is deterministic: identical input must
    /// produce identical sequence numbers in both engines. `ts_ns` is excluded
    /// because a wall clock is process-local by definition, so two engines can
    /// never agree on it and comparing it would make the test fail for a reason
    /// that says nothing about correctness.
    pub fn encode(&self) -> String {
        match *self {
            Event::Accepted {
                seq,
                id,
                side,
                price,
                qty,
                ..
            } => {
                let p = match price {
                    Some(v) => v.to_string(),
                    None => "M".to_string(),
                };
                format!("{} A {} {} {} {}", seq, id, side.as_char(), p, qty)
            }
            Event::Trade {
                seq,
                price,
                qty,
                aggressor,
                resting,
                ..
            } => format!("{} T {} {} {} {}", seq, price, qty, aggressor, resting),
            Event::Canceled { seq, id, remaining, .. } => {
                format!("{} C {} {}", seq, id, remaining)
            }
            Event::RejectCancel { seq, id, .. } => format!("{} R {}", seq, id),
        }
    }

    pub fn seq(&self) -> u64 {
        match *self {
            Event::Accepted { seq, .. }
            | Event::Trade { seq, .. }
            | Event::Canceled { seq, .. }
            | Event::RejectCancel { seq, .. } => seq,
        }
    }

    pub fn ts_ns(&self) -> u64 {
        match *self {
            Event::Accepted { ts_ns, .. }
            | Event::Trade { ts_ns, .. }
            | Event::Canceled { ts_ns, .. }
            | Event::RejectCancel { ts_ns, .. } => ts_ns,
        }
    }
}

/// Grow `v`'s capacity to `n` and write zeros over the unused part, so the
/// OS maps those pages now rather than mid-session. Writing into
/// MaybeUninit is safe; the length (what the Vec considers valid) is unchanged.
fn touch<T>(v: &mut Vec<T>, n: usize) {
    v.reserve(n.saturating_sub(v.len()));
    for slot in v.spare_capacity_mut() {
        *slot = std::mem::MaybeUninit::zeroed();
    }
}

/// One price level: a FIFO of slab slots (husks included) and how many of
/// them are live. The count is what stops a level of pure husks from
/// advertising a price nobody can trade at.
#[derive(Clone, Copy)]
struct Level {
    head: usize,
    tail: usize,
    live: i64,
}

impl Default for Level {
    fn default() -> Self {
        Level { head: NO_SLOT, tail: NO_SLOT, live: 0 }
    }
}

/// One side of the book: price -> level.
#[derive(Default)]
struct BookSide {
    levels: BTreeMap<Price, Level>,
}

pub struct Book {
    bids: BookSide,
    asks: BookSide,
    slab: Vec<Order>, // never shrinks; order id N is in slot N - 1
    pub events: Vec<Event>,
    next_id: OrderId,
    seq: u64,
    start: Instant,
    /// When set, timestamps come from the sequence number instead of the clock.
    /// Makes a log reproducible, which a wall clock can never be.
    deterministic_ts: bool,
}

impl Default for Book {
    fn default() -> Self {
        Self::new()
    }
}

impl Book {
    pub fn new() -> Self {
        Book {
            bids: BookSide::default(),
            asks: BookSide::default(),
            slab: Vec::with_capacity(1024),
            events: Vec::with_capacity(1024),
            next_id: 1,
            seq: 0,
            start: Instant::now(),
            deterministic_ts: false,
        }
    }

    /// Size the order slab and event log for a session up front, and
    /// touch the memory now. Without this the vectors double as they fill:
    /// every doubling copies the whole log (milliseconds, once it is large),
    /// and every fresh 4 KB page costs a page fault the first time it is
    /// written (microseconds, once every ~70 events). In the C++ port this
    /// took p99 from 4.4us to 0.74us. Only capacity changes: a book that
    /// outgrows the reservation still works, it just grows as before.
    pub fn reserve(&mut self, orders: usize, events: usize) {
        touch(&mut self.slab, orders);
        touch(&mut self.events, events);
    }

    /// Timestamps mirror the sequence number, so the log is byte-reproducible.
    pub fn deterministic() -> Self {
        let mut b = Self::new();
        b.deterministic_ts = true;
        b
    }

    /// Next (seq, ts_ns) pair. One clock read per event, which is what a real
    /// feed costs: measured at roughly 6% of throughput in the Python port.
    #[inline]
    fn stamp(&mut self) -> (u64, u64) {
        self.seq += 1;
        let ts = if self.deterministic_ts {
            self.seq
        } else {
            self.start.elapsed().as_nanos() as u64
        };
        (self.seq, ts)
    }

    pub fn best_bid(&self) -> Option<Price> {
        self.bids.levels.keys().next_back().copied()
    }

    pub fn best_ask(&self) -> Option<Price> {
        self.asks.levels.keys().next().copied()
    }

    /// Up to `levels` price levels on one side, best first, as (price, live qty).
    /// Live qty skips cancelled husks, so it is what a trader could actually hit.
    pub fn depth(&self, side: Side, levels: usize) -> Vec<(Price, Qty)> {
        match side {
            Side::Buy => self
                .bids
                .levels
                .iter()
                .rev()
                .take(levels)
                .map(|(&p, l)| (p, self.live_qty(l)))
                .collect(),
            Side::Sell => self
                .asks
                .levels
                .iter()
                .take(levels)
                .map(|(&p, l)| (p, self.live_qty(l)))
                .collect(),
        }
    }

    /// Slab slots in a level's queue, front to back, husks included.
    fn queue(&self, level: &Level) -> impl Iterator<Item = usize> + '_ {
        let mut slot = level.head;
        std::iter::from_fn(move || {
            if slot == NO_SLOT {
                return None;
            }
            let here = slot;
            slot = self.slab[here].next;
            Some(here)
        })
    }

    fn live_qty(&self, level: &Level) -> Qty {
        self.queue(level)
            .filter(|&slot| !self.slab[slot].cancelled)
            .map(|slot| self.slab[slot].remaining)
            .sum()
    }

    /// True while the order is resting and live (not filled, not cancelled).
    pub fn has_order(&self, id: OrderId) -> bool {
        id >= 1 && (id as usize) <= self.slab.len() && self.slab[id as usize - 1].resting
    }

    /// Total shares resting on the book (live orders only).
    pub fn resting_qty(&self) -> Qty {
        self.slab.iter().filter(|o| o.resting).map(|o| o.remaining).sum()
    }

    /// Ids of all resting live orders, ascending. Ids are issued in arrival
    /// order and an order rests at most once, so ascending id is also the order
    /// Python's insertion-ordered dict yields them in.
    // ponytail: O(every order ever) scan; only tests and the binding's
    // introspection call it.
    pub fn order_ids(&self) -> Vec<OrderId> {
        self.slab.iter().filter(|o| o.resting).map(|o| o.id).collect()
    }

    fn side_mut(&mut self, side: Side) -> &mut BookSide {
        match side {
            Side::Buy => &mut self.bids,
            Side::Sell => &mut self.asks,
        }
    }

    pub fn limit(&mut self, side: Side, price: Price, qty: Qty) -> OrderId {
        self.submit(side, Some(price), qty)
    }

    pub fn market(&mut self, side: Side, qty: Qty) -> OrderId {
        self.submit(side, None, qty)
    }

    fn submit(&mut self, side: Side, price: Option<Price>, qty: Qty) -> OrderId {
        assert!(qty > 0, "qty must be positive");
        let id = self.next_id;
        self.next_id += 1;

        let slot = self.slab.len();
        self.slab.push(Order {
            id,
            side,
            price,
            qty,
            remaining: qty,
            cancelled: false,
            resting: false,
            next: NO_SLOT,
        });
        let (seq, ts_ns) = self.stamp();
        self.events.push(Event::Accepted {
            seq,
            ts_ns,
            id,
            side,
            price,
            qty,
        });

        self.match_order(slot);

        let remaining = self.slab[slot].remaining;
        if remaining > 0 {
            match price {
                // A market order never rests; the unfilled part is cancelled.
                None => {
                    let (seq, ts_ns) = self.stamp();
                    self.events.push(Event::Canceled {
                        seq,
                        ts_ns,
                        id,
                        remaining,
                    });
                    self.slab[slot].remaining = 0;
                }
                Some(p) => {
                    let level = self.side_mut(side).levels.entry(p).or_default();
                    let old_tail = level.tail;
                    level.tail = slot;
                    level.live += 1;
                    if old_tail == NO_SLOT {
                        level.head = slot; // first order at this price
                    } else {
                        // `level` is no longer used, so `self` is free again
                        self.slab[old_tail].next = slot;
                    }
                    self.slab[slot].resting = true;
                }
            }
        }
        id
    }

    pub fn cancel(&mut self, id: OrderId) -> bool {
        if !self.has_order(id) {
            let (seq, ts_ns) = self.stamp();
            self.events.push(Event::RejectCancel { seq, ts_ns, id });
            return false;
        }
        let (side, price, remaining) = {
            let o = &mut self.slab[id as usize - 1];
            o.cancelled = true; // lazy: the husk stays in its queue until reached
            o.resting = false;
            (o.side, o.price, o.remaining)
        };
        if let Some(p) = price {
            let s = self.side_mut(side);
            if let Some(level) = s.levels.get_mut(&p) {
                level.live -= 1;
                if level.live <= 0 {
                    s.levels.remove(&p);
                }
            }
        }
        let (seq, ts_ns) = self.stamp();
        self.events.push(Event::Canceled { seq, ts_ns, id, remaining });
        true
    }

    fn crosses(&self, taker_price: Option<Price>, side: Side, best: Price) -> bool {
        match taker_price {
            None => true, // market order takes any price
            Some(p) => match side {
                Side::Buy => p >= best,
                Side::Sell => p <= best,
            },
        }
    }

    fn match_order(&mut self, slot: usize) {
        let side = self.slab[slot].side;
        let taker_price = self.slab[slot].price;
        let taker_id = self.slab[slot].id;
        let opp_side = match side {
            Side::Buy => Side::Sell,
            Side::Sell => Side::Buy,
        };

        loop {
            if self.slab[slot].remaining == 0 {
                break;
            }
            let best = match side {
                Side::Buy => self.best_ask(),
                Side::Sell => self.best_bid(),
            };
            let best = match best {
                Some(b) if self.crosses(taker_price, side, b) => b,
                _ => break,
            };

            // Copy the level out, work on it, write it back once: one tree
            // lookup per level instead of one per fill.
            let mut level = match self.side_mut(opp_side).levels.get(&best) {
                Some(l) => *l,
                None => break,
            };
            loop {
                if self.slab[slot].remaining == 0 || level.head == NO_SLOT {
                    break;
                }
                let resting_slot = level.head;

                // Husk left behind by a lazy cancel: drop it, do not fill it.
                if self.slab[resting_slot].cancelled {
                    level.head = self.slab[resting_slot].next;
                    continue;
                }

                let traded = self.slab[slot].remaining.min(self.slab[resting_slot].remaining);
                self.slab[slot].remaining -= traded;
                self.slab[resting_slot].remaining -= traded;
                let resting_id = self.slab[resting_slot].id;
                let (seq, ts_ns) = self.stamp();
                self.events.push(Event::Trade {
                    seq,
                    ts_ns,
                    price: best,
                    qty: traded,
                    aggressor: taker_id,
                    resting: resting_id,
                });

                if self.slab[resting_slot].remaining == 0 {
                    self.slab[resting_slot].resting = false;
                    level.head = self.slab[resting_slot].next;
                    level.live -= 1;
                }
            }
            if level.head == NO_SLOT {
                level.tail = NO_SLOT;
            }

            let s = self.side_mut(opp_side);
            if level.live <= 0 {
                s.levels.remove(&best);
            } else if let Some(l) = s.levels.get_mut(&best) {
                *l = level;
            }
            if level.live > 0 && self.slab[slot].remaining > 0 {
                // The queue only runs dry with qty left if a live order is
                // missing from it, which the invariants forbid. Without this
                // the outer loop would spin forever on the same level.
                panic!("live level {} with an empty queue", best);
            }
        }
    }

    /// Invariants that must hold after every operation. Used by the tests and
    /// by the `--check` mode of the binary.
    pub fn check(&self) {
        if let (Some(b), Some(a)) = (self.best_bid(), self.best_ask()) {
            assert!(b < a, "crossed book: {} >= {}", b, a);
        }
        let mut live_total = 0i64;
        for (side_name, s) in [("bid", &self.bids), ("ask", &self.asks)] {
            for (price, level) in s.levels.iter() {
                let mut actual = 0i64;
                for slot in self.queue(level) {
                    let o = &self.slab[slot];
                    assert_eq!(o.price, Some(*price), "{} order queued at the wrong price", side_name);
                    if !o.cancelled {
                        actual += 1;
                        assert!(o.resting && o.remaining > 0, "{} order {} live but not resting", side_name, o.id);
                    }
                }
                let claimed = level.live;
                assert_eq!(
                    claimed, actual,
                    "{} live count wrong at {}",
                    side_name, price
                );
                assert!(actual > 0, "{} level {} advertised with no live orders", side_name, price);
                live_total += actual;
            }
        }
        let flagged = self.slab.iter().filter(|o| o.resting).count() as i64;
        assert_eq!(live_total, flagged, "order flagged resting but not on the book");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// reserve() changes capacity only: the same ops give the same log with
    /// a reservation up front, mid-session, far too small, or none at all.
    #[test]
    fn reserve_changes_nothing() {
        let run = |at: usize, orders: usize, events: usize| {
            let mut b = Book::deterministic();
            let mut live: Vec<OrderId> = Vec::new();
            let mut s: u64 = 7;
            for i in 0..5_000usize {
                if i == at {
                    b.reserve(orders, events);
                }
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let r = s >> 33;
                let side = if r % 2 == 0 { Side::Buy } else { Side::Sell };
                let qty = (r % 100) as i64 + 1;
                match r % 10 {
                    0 | 1 => {
                        if let Some(id) = live.pop() {
                            b.cancel(id);
                        }
                    }
                    2 => {
                        b.market(side, qty);
                    }
                    _ => {
                        let off = (r % 12) as i64 + 1;
                        let price = if side == Side::Buy { 10_000 - off } else { 10_000 + off };
                        live.push(b.limit(side, price, qty));
                    }
                }
            }
            b.events.iter().map(|e| e.encode()).collect::<Vec<_>>()
        };
        let plain = run(usize::MAX, 0, 0);
        assert_eq!(run(0, 10_000, 20_000), plain, "reserve up front changed the log");
        assert_eq!(run(2_500, 10_000, 20_000), plain, "reserve mid-session changed the log");
        assert_eq!(run(0, 10, 10), plain, "outgrowing the reservation changed the log");
    }

    fn trades(b: &Book) -> Vec<(Price, Qty, OrderId, OrderId)> {
        b.events
            .iter()
            .filter_map(|e| match *e {
                Event::Trade {
                    price,
                    qty,
                    aggressor,
                    resting,
                    ..
                } => Some((price, qty, aggressor, resting)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn every_event_is_stamped_and_ordered() {
        let mut b = Book::new();
        b.limit(Side::Sell, 100, 5);
        b.limit(Side::Buy, 100, 5);
        let dead = b.limit(Side::Buy, 99, 5);
        b.cancel(dead);
        b.cancel(9999); // reject

        let seqs: Vec<u64> = b.events.iter().map(|e| e.seq()).collect();
        let expected: Vec<u64> = (1..=seqs.len() as u64).collect();
        assert_eq!(seqs, expected, "seq must be dense and monotonic");

        let ts: Vec<u64> = b.events.iter().map(|e| e.ts_ns()).collect();
        for w in ts.windows(2) {
            assert!(w[0] <= w[1], "timestamps went backwards");
        }
    }

    #[test]
    fn deterministic_mode_reproduces_the_log() {
        let run = || {
            let mut b = Book::deterministic();
            b.limit(Side::Sell, 100, 5);
            b.limit(Side::Buy, 100, 5);
            b.events.clone()
        };
        assert_eq!(run(), run(), "deterministic clock must reproduce exactly");
        assert_eq!(run()[0].ts_ns(), 1, "deterministic ts mirrors seq");
    }

    #[test]
    fn exact_fill_empties_the_book() {
        let mut b = Book::new();
        b.limit(Side::Sell, 100, 10);
        b.limit(Side::Buy, 100, 10);
        assert_eq!(trades(&b), vec![(100, 10, 2, 1)]);
        assert!(b.best_bid().is_none() && b.best_ask().is_none());
        b.check();
    }

    #[test]
    fn partial_fill_rests_the_remainder() {
        let mut b = Book::new();
        b.limit(Side::Sell, 100, 5);
        b.limit(Side::Buy, 100, 8);
        assert_eq!(b.best_bid(), Some(100));
        b.check();
    }

    #[test]
    fn fifo_within_a_price_level() {
        let mut b = Book::new();
        let first = b.limit(Side::Sell, 100, 5);
        b.limit(Side::Sell, 100, 5);
        b.limit(Side::Buy, 100, 5);
        assert_eq!(trades(&b)[0].3, first, "FIFO violated");
        b.check();
    }

    #[test]
    fn better_price_beats_earlier_time() {
        let mut b = Book::new();
        b.limit(Side::Sell, 101, 5);
        let cheap = b.limit(Side::Sell, 100, 5);
        b.limit(Side::Buy, 101, 5);
        assert_eq!(trades(&b)[0].3, cheap, "price priority violated");
        b.check();
    }

    #[test]
    fn fills_at_the_resting_price() {
        let mut b = Book::new();
        b.limit(Side::Sell, 100, 5);
        b.limit(Side::Buy, 105, 5);
        assert_eq!(trades(&b)[0].0, 100);
        b.check();
    }

    #[test]
    fn market_order_sweeps_and_never_rests() {
        let mut b = Book::new();
        b.limit(Side::Sell, 100, 3);
        b.limit(Side::Sell, 101, 3);
        b.market(Side::Buy, 10);
        let qtys: Vec<Qty> = trades(&b).iter().map(|t| t.1).collect();
        assert_eq!(qtys, vec![3, 3]);
        assert!(b.best_bid().is_none(), "market order must not rest");
        b.check();
    }

    #[test]
    fn cancel_then_double_cancel() {
        let mut b = Book::new();
        let oid = b.limit(Side::Sell, 100, 10);
        b.limit(Side::Buy, 100, 4);
        assert!(b.cancel(oid));
        assert!(b.best_ask().is_none());
        assert!(!b.cancel(oid), "double cancel must fail");
        b.check();
    }

    #[test]
    fn fully_cancelled_level_stops_advertising() {
        let mut b = Book::new();
        let a = b.limit(Side::Buy, 100, 5);
        let c = b.limit(Side::Buy, 100, 5);
        b.cancel(a);
        assert_eq!(b.best_bid(), Some(100));
        b.cancel(c);
        assert!(b.best_bid().is_none(), "phantom level left behind");
        b.check();
    }

    #[test]
    fn cancelled_order_is_never_filled() {
        let mut b = Book::new();
        let dead = b.limit(Side::Sell, 100, 5);
        let alive = b.limit(Side::Sell, 100, 5);
        b.cancel(dead);
        b.limit(Side::Buy, 100, 5);
        let t = trades(&b);
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].3, alive, "filled a cancelled order");
        b.check();
    }

    #[test]
    fn portable_accessors() {
        let mut b = Book::new();
        let a = b.limit(Side::Buy, 99, 5);
        let c = b.limit(Side::Buy, 99, 7);
        b.limit(Side::Buy, 98, 1);
        b.limit(Side::Sell, 101, 2);
        b.cancel(a);
        assert_eq!(b.depth(Side::Buy, 5), vec![(99, 7), (98, 1)]);
        assert_eq!(b.depth(Side::Buy, 1), vec![(99, 7)]);
        assert_eq!(b.depth(Side::Sell, 5), vec![(101, 2)]);
        assert!(!b.has_order(a) && b.has_order(c));
        assert_eq!(b.order_ids(), vec![2, 3, 4]);
    }

    /// Deterministic LCG so the sequence does not depend on a crate.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn range(&mut self, lo: i64, hi: i64) -> i64 {
            lo + (self.next() % ((hi - lo + 1) as u64)) as i64
        }
    }

    #[test]
    fn randomized_stream_conserves_shares() {
        let mut rng = Lcg(42);
        let mut b = Book::new();
        let mut submitted: i64 = 0;
        let mut live_ids: Vec<OrderId> = Vec::new();

        for _ in 0..20_000 {
            let side = if rng.next() % 2 == 0 { Side::Buy } else { Side::Sell };
            let roll = rng.range(0, 99);
            if roll < 15 && !live_ids.is_empty() {
                let i = (rng.next() as usize) % live_ids.len();
                let id = live_ids.swap_remove(i);
                b.cancel(id);
            } else if roll < 30 {
                let q = rng.range(1, 20);
                submitted += q;
                b.market(side, q);
            } else {
                let q = rng.range(1, 20);
                submitted += q;
                let id = b.limit(side, rng.range(95, 105), q);
                live_ids.push(id);
            }
            b.check();
        }

        let mut filled: i64 = 0;
        let mut canceled: i64 = 0;
        for e in &b.events {
            match *e {
                Event::Trade { qty, .. } => filled += qty,
                Event::Canceled { remaining, .. } => canceled += remaining,
                _ => {}
            }
        }
        let resting: i64 = b.resting_qty();

        // Each fill consumes `traded` from BOTH sides, so a trade of size q
        // accounts for 2q of the submitted shares.
        assert_eq!(
            2 * filled + canceled + resting,
            submitted,
            "shares leaked: 2*{} + {} + {} != {}",
            filled,
            canceled,
            resting,
            submitted
        );
    }
}
