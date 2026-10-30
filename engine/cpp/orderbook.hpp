// Limit order book matching engine, price-time priority. C++17, std only.
//
// This is the third implementation of the same engine, after the Python
// oracle (engine/orderbook.py) and the Rust port (engine/rust/src/lib.rs).
// It is a port, not a redesign: engine/diff_test.py feeds all three the same
// op file and requires byte-identical event logs. A fast engine that is subtly
// different is worth nothing, because every fill built on it is quietly wrong.
//
// Matching rules (what real exchanges do):
//   1. Better price wins.
//   2. At the same price, whoever arrived first wins (FIFO).
// An incoming order sweeps the opposite side while prices cross, filling at
// the RESTING order's price (the resting order set the terms), and rests what
// remains. Market orders never rest: the unfilled remainder is cancelled.
//
// Prices are integer ticks and quantities are integer shares, never floats:
// 0.1 + 0.2 != 0.3, and money bugs from float rounding are silent.
//
// Data structures follow the Rust design, except the id lookup and queues:
//   * price levels:  std::map<Price, Level>, an ordered tree, so the best bid
//     is the last key and the best ask the first key (O(1) via begin/rbegin,
//     O(log n) insert/erase).
//   * order storage: one flat std::vector<Order> (a slab), referenced by
//     index, so orders are contiguous instead of individually heap-allocated.
//   * id lookup:     none needed. Ids are issued 1, 2, 3... and every order
//     takes the next slab slot, so order `id` lives in slot id - 1. Cancel is
//     an array index, not a hash lookup.
//   * per-level FIFO: an intrusive singly linked list threaded through the
//     slab (Order::next), with head/tail in the Level. Joining or leaving a
//     queue never allocates; a std::deque allocated a 512-byte chunk for
//     every new price level.
//   * cancellation:  lazy. A cancel flags the order and decrements the level's
//     live counter; matching pops the flagged "husk" when it reaches it. A
//     level whose live count hits 0 is erased at once, which is why the
//     counter exists: otherwise a level of pure husks would advertise a price.
//
// One deliberate, behaviour-neutral deviation from Rust: Rust keeps two
// parallel maps per side (price -> queue and price -> live count) and always
// inserts and removes them together. Here they are one map of
// Level{head, tail, live}, which halves the tree lookups and makes it impossible
// for the two to drift out of sync.
//
// Invariants (checked by Book::check_invariants, which throws on violation):
//   I1  best bid < best ask whenever both exist (the book is never crossed).
//   I2  every advertised level has live > 0, and live equals the number of
//       non-cancelled orders in its FIFO.
//   I3  every live resting order has remaining > 0 and is flagged resting;
//       nothing else is flagged resting.
//   I4  seq is dense: the n-th event has seq n.

#pragma once

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace bullpen {

using Price = std::int64_t;     // integer ticks; dollars = ticks * tick_size
using Qty = std::int64_t;       // whole shares
using OrderId = std::uint64_t;  // issued sequentially from 1

// The underlying char is the wire letter, so encoding a side is a cast.
enum class Side : char { Buy = 'B', Sell = 'S' };

[[nodiscard]] constexpr char side_char(Side s) noexcept { return static_cast<char>(s); }

[[nodiscard]] constexpr bool is_valid(Side s) noexcept {
    return s == Side::Buy || s == Side::Sell;
}

[[nodiscard]] constexpr Side opposite(Side s) noexcept {
    return s == Side::Buy ? Side::Sell : Side::Buy;
}

inline constexpr std::size_t kNoSlot = static_cast<std::size_t>(-1);

struct Order {
    OrderId id;
    Side side;
    std::optional<Price> price;  // nullopt = market order
    Qty qty;
    Qty remaining;
    bool cancelled;
    bool resting = false;        // live on the book: what has_order() reports
    std::size_t next = kNoSlot;  // next order in the same level's FIFO
};

// ---- events -------------------------------------------------------------
// Same set and field order as Rust's `Event` enum. Every event carries a
// monotonic `seq` and a timestamp, the way a real venue's feed does (ITCH
// stamps every message). `seq` is the authoritative ordering because it is
// deterministic and survives replay; `ts_ns` is process-local, so it is not
// part of encode().
namespace ev {
struct Accepted {
    std::uint64_t seq;
    std::uint64_t ts_ns;
    OrderId id;
    Side side;
    std::optional<Price> price;  // nullopt = market order, encoded as "M"
    Qty qty;
};
struct Trade {
    std::uint64_t seq;
    std::uint64_t ts_ns;
    Price price;  // the resting order's price
    Qty qty;
    OrderId aggressor;
    OrderId resting;
};
struct Canceled {
    std::uint64_t seq;
    std::uint64_t ts_ns;
    OrderId id;
    Qty remaining;  // shares that will now never trade
};
struct RejectCancel {
    std::uint64_t seq;
    std::uint64_t ts_ns;
    OrderId id;
};
}  // namespace ev

// A closed sum type, like the Rust enum: std::visit must handle every case.
using Event = std::variant<ev::Accepted, ev::Trade, ev::Canceled, ev::RejectCancel>;

[[nodiscard]] inline std::uint64_t seq_of(const Event& e) noexcept {
    return std::visit([](const auto& x) noexcept { return x.seq; }, e);
}

[[nodiscard]] inline std::uint64_t ts_of(const Event& e) noexcept {
    return std::visit([](const auto& x) noexcept { return x.ts_ns; }, e);
}

namespace detail {
// std::to_chars: locale-free, allocation-free, exact for integers. Faster and
// more predictable than ostream or snprintf, and portable across GCC/MSVC.
template <typename Int>
void append_int(std::string& out, Int v) {
    char buf[24];  // 20 digits for uint64 max, plus sign
    const auto res = std::to_chars(buf, buf + sizeof buf, v);
    out.append(buf, res.ptr);
}
}  // namespace detail

// Wire form used by the differential test. Must match encode() in
// engine/diff_test.py and Event::encode in engine/rust/src/lib.rs exactly:
//   "<seq> A <id> <B|S> <price|M> <qty>"
//   "<seq> T <price> <qty> <aggressor> <resting>"
//   "<seq> C <id> <remaining>"
//   "<seq> R <id>"
// Appends without a newline, so a caller can build one big buffer.
inline void append_encoded(std::string& out, const Event& e) {
    std::visit(
        [&out](const auto& x) {
            using T = std::decay_t<decltype(x)>;
            detail::append_int(out, x.seq);
            if constexpr (std::is_same_v<T, ev::Accepted>) {
                out += " A ";
                detail::append_int(out, x.id);
                out += ' ';
                out += side_char(x.side);
                out += ' ';
                if (x.price) {
                    detail::append_int(out, *x.price);
                } else {
                    out += 'M';
                }
                out += ' ';
                detail::append_int(out, x.qty);
            } else if constexpr (std::is_same_v<T, ev::Trade>) {
                out += " T ";
                detail::append_int(out, x.price);
                out += ' ';
                detail::append_int(out, x.qty);
                out += ' ';
                detail::append_int(out, x.aggressor);
                out += ' ';
                detail::append_int(out, x.resting);
            } else if constexpr (std::is_same_v<T, ev::Canceled>) {
                out += " C ";
                detail::append_int(out, x.id);
                out += ' ';
                detail::append_int(out, x.remaining);
            } else {
                static_assert(std::is_same_v<T, ev::RejectCancel>, "unhandled event type");
                out += " R ";
                detail::append_int(out, x.id);
            }
        },
        e);
}

[[nodiscard]] inline std::string encode(const Event& e) {
    std::string s;
    append_encoded(s, e);
    return s;
}

// ---- the book -------------------------------------------------------------
class Book {
public:
    // Steady: ts_ns = nanoseconds since the book was created (monotonic, so it
    // cannot run backwards mid-session). Deterministic: ts_ns = seq, which
    // makes a log byte-reproducible; a wall clock never can be.
    enum class Clock { Steady, Deterministic };

    explicit Book(Clock clock = Clock::Steady)
        : clock_(clock), start_(std::chrono::steady_clock::now()) {
        slab_.reserve(1024);
        events_.reserve(1024);
    }

    // Size the order slab and event log for a session up front, and touch
    // the memory now. Without this, the vectors double as they fill: every
    // doubling copies the whole log (milliseconds, once the log is large),
    // and every fresh 4 KB page costs a page fault the first time it is
    // written (microseconds, once every ~70 events). Real venues size their
    // memory at startup for the same reason. Only capacity changes: a book
    // that outgrows the reservation still works, it just grows as before.
    void reserve(std::size_t orders, std::size_t events) {
        slab_.resize(std::max(orders, slab_.size()));  // resize writes every page...
        slab_.resize(next_id_ - 1);                    // ...then drop back; capacity stays
        events_.resize(std::max(events, events_.size()));
        events_.resize(seq_);
    }

    [[nodiscard]] static Book deterministic() { return Book(Clock::Deterministic); }

    // ---- commands ---------------------------------------------------------
    // limit/market return the new order id but are not [[nodiscard]]: a
    // fire-and-forget market order is legitimate, and the id is also in the
    // accepted event.

    // Throws std::invalid_argument (and changes nothing: no event, no id
    // consumed) on a bad side, non-positive qty or non-positive price, like
    // the Python oracle's ValueError.
    OrderId limit(Side side, Price price, Qty qty) { return submit(side, price, qty); }

    OrderId market(Side side, Qty qty) { return submit(side, std::nullopt, qty); }

    // Python refuses a float price or qty even when it is whole (100.0),
    // because it means the caller did money arithmetic in floating point
    // upstream. C++ would silently truncate a double to int64 here, so the
    // same rule is enforced at compile time: any non-integer argument type
    // selects these deleted overloads instead of converting.
    template <typename P, typename Q,
              std::enable_if_t<!(std::is_integral_v<P> && std::is_integral_v<Q>), int> = 0>
    OrderId limit(Side, P, Q) = delete;
    template <typename Q, std::enable_if_t<!std::is_integral_v<Q>, int> = 0>
    OrderId market(Side, Q) = delete;

    // false (plus a reject_cancel event) for an id that was never issued, has
    // already filled, or was already cancelled. [[nodiscard]] because ignoring
    // a failed cancel is a real trading bug: the order may have filled, and a
    // caller who assumes it is gone now holds a position it doesn't know
    // about. A replay loop that only wants the event log says so with a void
    // cast.
    [[nodiscard]] bool cancel(OrderId id) {
        if (!has_order(id)) {
            const auto [seq, ts] = stamp();
            events_.emplace_back(ev::RejectCancel{seq, ts, id});
            return false;
        }
        Order& o = slab_[id - 1];
        o.resting = false;
        o.cancelled = true;  // lazy: the husk stays in its FIFO until reached
        if (o.price) {
            Levels& side = levels(o.side);
            const auto lvl = side.find(*o.price);
            // Always found for a resting order (I2/I3); checked, not assumed.
            if (lvl != side.end() && --lvl->second.live <= 0) {
                side.erase(lvl);  // no live orders left: stop advertising it
            }
        }
        const auto [seq, ts] = stamp();
        events_.emplace_back(ev::Canceled{seq, ts, id, o.remaining});
        return true;
    }

    // ---- introspection (the portable interface, SPEC.md 7.1) --------------
    [[nodiscard]] std::optional<Price> best_bid() const noexcept {
        if (bids_.empty()) return std::nullopt;
        return bids_.rbegin()->first;
    }

    [[nodiscard]] std::optional<Price> best_ask() const noexcept {
        if (asks_.empty()) return std::nullopt;
        return asks_.begin()->first;
    }

    // Up to `max_levels` levels, best first, as (price, live qty). Live qty
    // skips husks, so it is what a trader could actually hit.
    [[nodiscard]] std::vector<std::pair<Price, Qty>> depth(Side side,
                                                           std::size_t max_levels = 5) const {
        std::vector<std::pair<Price, Qty>> out;
        const auto take = [&](auto first, auto last) {
            for (; first != last && out.size() < max_levels; ++first) {
                out.emplace_back(first->first, live_qty(first->second));
            }
        };
        if (side == Side::Buy) {
            take(bids_.rbegin(), bids_.rend());
        } else {
            take(asks_.begin(), asks_.end());
        }
        return out;
    }

    [[nodiscard]] std::size_t event_count() const noexcept { return events_.size(); }

    [[nodiscard]] const std::vector<Event>& events() const noexcept { return events_; }

    // True while the order is resting and live (not filled, not cancelled).
    [[nodiscard]] bool has_order(OrderId id) const noexcept {
        return id >= 1 && id <= slab_.size() && slab_[id - 1].resting;
    }

    // Live resting ids, ascending. Ids are issued in arrival order and an
    // order rests at most once, so this matches Python's dict order.
    // ponytail: O(every order ever) scan; only tests and replays call it.
    [[nodiscard]] std::vector<OrderId> order_ids() const {
        std::vector<OrderId> ids;
        for (const Order& o : slab_) {
            if (o.resting) ids.push_back(o.id);
        }
        return ids;
    }

    // Total shares resting on the book (live orders only). Used by the share
    // conservation check: submitted == 2*filled + canceled + resting.
    [[nodiscard]] Qty resting_qty() const {
        Qty total = 0;
        for (const Order& o : slab_) {
            if (o.resting) total += o.remaining;
        }
        return total;
    }

    // Throws std::logic_error naming the first broken invariant. Not an
    // assert(): it must still run in the -DNDEBUG release build that replay
    // uses, because the differential test relies on it.
    // O(resting orders + husks): fine per op in tests, quadratic over a long
    // replay, which is why check_touch() exists.
    void check_invariants() const {
        check_touch();

        std::size_t live_total = 0;
        for (const Levels* side : {&bids_, &asks_}) {
            for (const auto& [price, level] : *side) {
                std::int64_t actual = 0;
                for (std::size_t slot = level.head; slot != kNoSlot; slot = slab_[slot].next) {
                    const Order& o = slab_[slot];
                    if (!o.price || *o.price != price) fail("I2 order queued at the wrong price");
                    if (o.cancelled) continue;
                    ++actual;
                    if (o.remaining <= 0) fail("I3 fully filled order still resting");
                    if (!o.resting || o.id != slot + 1) fail("I3 resting order not flagged");
                }
                if (level.live != actual) fail("I2 live count out of sync");
                if (actual <= 0) fail("I2 level advertised with no live orders");
                live_total += static_cast<std::size_t>(actual);
            }
        }
        std::size_t flagged = 0;
        for (const Order& o : slab_) flagged += o.resting ? 1 : 0;
        if (live_total != flagged) fail("I3 order flagged resting but not on the book");
    }

    // The O(1) subset of check_invariants: I1 and I4.
    void check_touch() const {
        const auto bb = best_bid();
        const auto ba = best_ask();
        if (bb && ba && *bb >= *ba) fail("I1 crossed book");
        if (seq_ != events_.size()) fail("I4 seq not dense");
    }

private:
    struct Level {
        std::size_t head = kNoSlot;  // oldest slab slot in the queue, husks included
        std::size_t tail = kNoSlot;  // newest, where arrivals join
        std::int64_t live = 0;       // non-cancelled orders in the queue
    };
    using Levels = std::map<Price, Level>;

    [[noreturn]] static void fail(const char* what) {
        throw std::logic_error(std::string("order book invariant broken: ") + what);
    }

    Levels& levels(Side s) noexcept { return s == Side::Buy ? bids_ : asks_; }

    [[nodiscard]] Qty live_qty(const Level& level) const {
        Qty q = 0;
        for (std::size_t slot = level.head; slot != kNoSlot; slot = slab_[slot].next) {
            if (!slab_[slot].cancelled) q += slab_[slot].remaining;
        }
        return q;
    }

    // Next (seq, ts_ns). One clock read per event, which is what a real feed
    // pays; the deterministic mode skips it.
    std::pair<std::uint64_t, std::uint64_t> stamp() {
        ++seq_;
        if (clock_ == Clock::Deterministic) return {seq_, seq_};
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - start_)
                            .count();
        return {seq_, static_cast<std::uint64_t>(ns)};  // steady: never negative
    }

    OrderId submit(Side side, std::optional<Price> price, Qty qty) {
        // Validate before touching any state, so a rejected order leaves no
        // residue: no event, no id consumed (matches Python's ValueError).
        if (!is_valid(side)) throw std::invalid_argument("bad order: side must be B or S");
        if (qty <= 0) throw std::invalid_argument("bad order: qty must be positive");
        if (price && *price <= 0) {
            throw std::invalid_argument("price must be a positive whole number of ticks");
        }

        const OrderId id = next_id_++;
        const std::size_t slot = slab_.size();
        slab_.push_back(Order{id, side, price, qty, qty, false});
        {
            const auto [seq, ts] = stamp();
            events_.emplace_back(ev::Accepted{seq, ts, id, side, price, qty});
        }

        match(slot);

        Order& o = slab_[slot];
        if (o.remaining > 0) {
            if (!price) {
                // A market order never rests; the unfilled part is cancelled.
                const auto [seq, ts] = stamp();
                events_.emplace_back(ev::Canceled{seq, ts, id, o.remaining});
                o.remaining = 0;
            } else {
                Level& level = levels(side)[*price];  // creates the level if new
                if (level.tail == kNoSlot) {
                    level.head = slot;
                } else {
                    slab_[level.tail].next = slot;
                }
                level.tail = slot;
                ++level.live;
                o.resting = true;
            }
        }
        return id;
    }

    void pop_front(Level& level) noexcept {
        level.head = slab_[level.head].next;
        if (level.head == kNoSlot) level.tail = kNoSlot;
    }

    [[nodiscard]] static bool crosses(const Order& taker, Price best) noexcept {
        if (!taker.price) return true;  // market order takes any price
        return taker.side == Side::Buy ? *taker.price >= best : *taker.price <= best;
    }

    void match(std::size_t slot) {
        // slab_ does not grow while matching, so this reference stays valid.
        Order& taker = slab_[slot];
        Levels& opp = levels(opposite(taker.side));

        while (taker.remaining > 0 && !opp.empty()) {
            // A buyer hits the lowest ask; a seller hits the highest bid.
            const auto it = taker.side == Side::Buy ? opp.begin() : std::prev(opp.end());
            const Price best = it->first;
            if (!crosses(taker, best)) break;

            Level& level = it->second;
            while (taker.remaining > 0 && level.head != kNoSlot) {
                Order& resting = slab_[level.head];
                if (resting.cancelled) {  // husk left by a lazy cancel: drop, never fill
                    pop_front(level);
                    continue;
                }
                const Qty traded = std::min(taker.remaining, resting.remaining);
                taker.remaining -= traded;
                resting.remaining -= traded;
                const auto [seq, ts] = stamp();
                events_.emplace_back(ev::Trade{seq, ts, best, traded, taker.id, resting.id});

                if (resting.remaining == 0) {
                    resting.resting = false;
                    pop_front(level);
                    --level.live;
                }
            }

            if (level.live <= 0) {
                opp.erase(it);
            } else if (taker.remaining > 0) {
                // The inner loop only stops with qty left if the FIFO ran dry,
                // which I2 forbids while live > 0. Without this the outer loop
                // would spin forever on the same level.
                fail("I2 live level with an empty queue");
            }
        }
    }

    Clock clock_;
    std::chrono::steady_clock::time_point start_;
    Levels bids_;
    Levels asks_;
    std::vector<Order> slab_;  // never shrinks; order id N is in slot N - 1
    std::vector<Event> events_;
    OrderId next_id_ = 1;
    std::uint64_t seq_ = 0;
};

}  // namespace bullpen
