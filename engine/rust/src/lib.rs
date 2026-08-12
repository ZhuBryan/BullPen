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
//!   by index. Ids map to slots through a `HashMap`, so cancelling is O(1) and
//!   the orders themselves stay contiguous in memory instead of being scattered
//!   across individually allocated objects.
//!
//! * CANCELLATION. Same lazy scheme as Python: cancelling flags the order and
//!   decrements a per-level live counter, and matching skips the husks. The
//!   counter is what stops a level of pure husks from advertising a price.
//!
//! Prices are integer ticks. Never floats: 0.1 + 0.2 != 0.3 and money bugs from
//! float rounding are silent.

use std::collections::{BTreeMap, HashMap, VecDeque};

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
}

/// Everything the engine emits. Formatted identically to the Python side so the
/// two logs can be diffed directly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    Accepted {
        id: OrderId,
        side: Side,
        price: Option<Price>,
        qty: Qty,
    },
    Trade {
        price: Price,
        qty: Qty,
        aggressor: OrderId,
        resting: OrderId,
    },
    Canceled {
        id: OrderId,
        remaining: Qty,
    },
    RejectCancel {
        id: OrderId,
    },
}

impl Event {
    pub fn encode(&self) -> String {
        match *self {
            Event::Accepted {
                id,
                side,
                price,
                qty,
            } => {
                let p = match price {
                    Some(v) => v.to_string(),
                    None => "M".to_string(),
                };
                format!("A {} {} {} {}", id, side.as_char(), p, qty)
            }
            Event::Trade {
                price,
                qty,
                aggressor,
                resting,
            } => format!("T {} {} {} {}", price, qty, aggressor, resting),
            Event::Canceled { id, remaining } => format!("C {} {}", id, remaining),
            Event::RejectCancel { id } => format!("R {}", id),
        }
    }
}

/// One side of the book: price -> FIFO queue of slab indices, plus a live count.
#[derive(Default)]
struct BookSide {
    levels: BTreeMap<Price, VecDeque<usize>>,
    live: BTreeMap<Price, i64>,
}

pub struct Book {
    bids: BookSide,
    asks: BookSide,
    slab: Vec<Order>,
    index: HashMap<OrderId, usize>, // id -> slab slot, for O(1) cancel
    pub events: Vec<Event>,
    next_id: OrderId,
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
            index: HashMap::with_capacity(1024),
            events: Vec::with_capacity(1024),
            next_id: 1,
        }
    }

    pub fn best_bid(&self) -> Option<Price> {
        self.bids.levels.keys().next_back().copied()
    }

    pub fn best_ask(&self) -> Option<Price> {
        self.asks.levels.keys().next().copied()
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
        });
        self.events.push(Event::Accepted {
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
                    self.events.push(Event::Canceled {
                        id,
                        remaining,
                    });
                    self.slab[slot].remaining = 0;
                }
                Some(p) => {
                    let s = self.side_mut(side);
                    s.levels.entry(p).or_default().push_back(slot);
                    *s.live.entry(p).or_insert(0) += 1;
                    self.index.insert(id, slot);
                }
            }
        }
        id
    }

    pub fn cancel(&mut self, id: OrderId) -> bool {
        let slot = match self.index.remove(&id) {
            Some(s) => s,
            None => {
                self.events.push(Event::RejectCancel { id });
                return false;
            }
        };
        let (side, price, remaining) = {
            let o = &mut self.slab[slot];
            o.cancelled = true;
            (o.side, o.price, o.remaining)
        };
        if let Some(p) = price {
            let s = self.side_mut(side);
            if let Some(c) = s.live.get_mut(&p) {
                *c -= 1;
                if *c <= 0 {
                    s.levels.remove(&p);
                    s.live.remove(&p);
                }
            }
        }
        self.events.push(Event::Canceled { id, remaining });
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

            loop {
                if self.slab[slot].remaining == 0 {
                    break;
                }
                let resting_slot = {
                    let s = self.side_mut(opp_side);
                    match s.levels.get_mut(&best).and_then(|q| q.front().copied()) {
                        Some(rs) => rs,
                        None => break,
                    }
                };

                // Husk left behind by a lazy cancel: drop it, do not fill it.
                if self.slab[resting_slot].cancelled {
                    if let Some(q) = self.side_mut(opp_side).levels.get_mut(&best) {
                        q.pop_front();
                    }
                    continue;
                }

                let traded = self.slab[slot].remaining.min(self.slab[resting_slot].remaining);
                self.slab[slot].remaining -= traded;
                self.slab[resting_slot].remaining -= traded;
                let resting_id = self.slab[resting_slot].id;
                self.events.push(Event::Trade {
                    price: best,
                    qty: traded,
                    aggressor: taker_id,
                    resting: resting_id,
                });

                if self.slab[resting_slot].remaining == 0 {
                    if let Some(q) = self.side_mut(opp_side).levels.get_mut(&best) {
                        q.pop_front();
                    }
                    let s = self.side_mut(opp_side);
                    if let Some(c) = s.live.get_mut(&best) {
                        *c -= 1;
                    }
                    self.index.remove(&resting_id);
                }
            }

            let empty = {
                let s = self.side_mut(opp_side);
                s.live.get(&best).copied().unwrap_or(0) <= 0
            };
            if empty {
                let s = self.side_mut(opp_side);
                s.levels.remove(&best);
                s.live.remove(&best);
            }
        }
    }

    /// Invariants that must hold after every operation. Used by the tests and
    /// by the `--check` mode of the binary.
    pub fn check(&self) {
        if let (Some(b), Some(a)) = (self.best_bid(), self.best_ask()) {
            assert!(b < a, "crossed book: {} >= {}", b, a);
        }
        for (side_name, s) in [("bid", &self.bids), ("ask", &self.asks)] {
            for (price, q) in s.levels.iter() {
                let actual = q
                    .iter()
                    .filter(|&&slot| !self.slab[slot].cancelled)
                    .count() as i64;
                let claimed = s.live.get(price).copied().unwrap_or(0);
                assert_eq!(
                    claimed, actual,
                    "{} live count wrong at {}",
                    side_name, price
                );
                assert!(actual > 0, "{} level {} advertised with no live orders", side_name, price);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trades(b: &Book) -> Vec<(Price, Qty, OrderId, OrderId)> {
        b.events
            .iter()
            .filter_map(|e| match *e {
                Event::Trade {
                    price,
                    qty,
                    aggressor,
                    resting,
                } => Some((price, qty, aggressor, resting)),
                _ => None,
            })
            .collect()
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
        let resting: i64 = b
            .index
            .values()
            .map(|&slot| b.slab[slot].remaining)
            .sum();

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
