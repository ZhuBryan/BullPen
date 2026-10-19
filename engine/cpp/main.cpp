// Command-line harness for the C++ engine. Mirrors engine/rust/src/main.rs.
//
//   bullpen_cpp replay <opsfile>     print the event log, one encoded line per
//                                    event, for engine/diff_test.py to compare
//   bullpen_cpp bench [opsfile|N]    throughput + tail latency (default: 200,000
//                                    generated ops)
//   bullpen_cpp selftest             behavioural tests; exit code 1 on failure
//
// The benchmark deliberately mirrors bench.py and the Rust bench: warmup
// discarded, throughput measured as wall time over N ops (no per-op timer
// overhead), latency percentiles measured per op with the timer's own cost
// reported alongside, and a fixed seed or a shared op file so every engine
// runs identical input. Comparing two engines on different input proves
// nothing.
//
// Op-file format, one op per line (shared with diff_test.py and Rust):
//   L <B|S> <price> <qty>    limit order
//   M <B|S> <qty>            market order
//   X <id>                   cancel
// Blank lines and lines starting with '#' are ignored.

#include "orderbook.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace {

using namespace bullpen;
using SteadyClock = std::chrono::steady_clock;

// ---- ops and parsing -------------------------------------------------------

struct Op {
    enum class Kind : std::uint8_t {
        Limit,
        Market,
        Cancel,    // cancel whatever this run most recently rested (generated streams)
        CancelId,  // cancel a specific id (op files)
    };
    Kind kind;
    Side side;
    Price price;
    Qty qty;
    OrderId id;

    [[nodiscard]] static Op limit(Side s, Price p, Qty q) { return {Kind::Limit, s, p, q, 0}; }
    [[nodiscard]] static Op market(Side s, Qty q) { return {Kind::Market, s, 0, q, 0}; }
    [[nodiscard]] static Op cancel_last() { return {Kind::Cancel, Side::Buy, 0, 0, 0}; }
    [[nodiscard]] static Op cancel_id(OrderId id) { return {Kind::CancelId, Side::Buy, 0, 0, id}; }
};

[[nodiscard]] constexpr bool is_space(char c) noexcept {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\v' || c == '\f';
}

// Whole-token integer parse. std::from_chars accepts no '+', no whitespace, no
// decimal point and no exponent, and reports overflow, so "100.0", "1e3" and
// "99999999999999999999" are all refused here rather than truncated. That is
// the input-side half of the engines' "no floats" rule.
template <typename Int>
[[nodiscard]] Int parse_int(std::string_view tok, const char* what) {
    Int v{};
    const char* const first = tok.data();
    const char* const last = first + tok.size();
    const auto res = std::from_chars(first, last, v);
    if (res.ec != std::errc{} || res.ptr != last) {
        throw std::invalid_argument(std::string(what) + " must be a whole number in range, got '" +
                                    std::string(tok) + "'");
    }
    return v;
}

[[nodiscard]] Side parse_side(std::string_view tok) {
    if (tok == "B") return Side::Buy;
    if (tok == "S") return Side::Sell;
    throw std::invalid_argument("side must be B or S, got '" + std::string(tok) + "'");
}

// One line -> one op, or nullopt for a blank/comment line. Throws
// std::invalid_argument on anything malformed.
//
// Deliberately stricter than the Python and Rust readers, which treat any side
// other than "B" as a sell and ignore trailing fields. Neither case occurs in
// generated files, so the diff is unaffected; a hand-edited file with a typo
// fails loudly here instead of trading on the wrong side.
[[nodiscard]] std::optional<Op> parse_line(std::string_view line) {
    std::array<std::string_view, 4> f{};
    std::size_t n = 0;
    std::size_t i = 0;
    while (i < line.size()) {
        while (i < line.size() && is_space(line[i])) ++i;
        if (i == line.size()) break;
        const std::size_t start = i;
        while (i < line.size() && !is_space(line[i])) ++i;
        if (n == f.size()) throw std::invalid_argument("too many fields");
        f[n++] = line.substr(start, i - start);
    }
    if (n == 0 || f[0].front() == '#') return std::nullopt;

    const auto need = [n](std::size_t k) {
        if (n != k) throw std::invalid_argument("wrong number of fields");
    };
    if (f[0] == "L") {
        need(4);
        return Op::limit(parse_side(f[1]), parse_int<Price>(f[2], "price"),
                         parse_int<Qty>(f[3], "qty"));
    }
    if (f[0] == "M") {
        need(3);
        return Op::market(parse_side(f[1]), parse_int<Qty>(f[2], "qty"));
    }
    if (f[0] == "X") {
        need(2);
        return Op::cancel_id(parse_int<OrderId>(f[1], "order id"));
    }
    throw std::invalid_argument("unknown op '" + std::string(f[0]) + "'");
}

[[nodiscard]] std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("could not read op file " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// Calls f(line_number, line) for every line, 1-based, without copying.
template <typename F>
void for_each_line(std::string_view text, F&& f) {
    std::size_t line_no = 0;
    std::size_t pos = 0;
    while (pos < text.size()) {
        std::size_t nl = text.find('\n', pos);
        if (nl == std::string_view::npos) nl = text.size();
        f(++line_no, text.substr(pos, nl - pos));
        pos = nl + 1;
    }
}

[[noreturn]] void rethrow_at(const std::string& path, std::size_t line_no,
                             const std::exception& e) {
    throw std::runtime_error(path + ":" + std::to_string(line_no) + ": " + e.what());
}

[[nodiscard]] std::vector<Op> load_ops(const std::string& path) {
    const std::string text = read_file(path);
    std::vector<Op> ops;
    for_each_line(text, [&](std::size_t line_no, std::string_view line) {
        try {
            if (auto op = parse_line(line)) ops.push_back(*op);
        } catch (const std::exception& e) {
            rethrow_at(path, line_no, e);
        }
    });
    return ops;
}

// One op against the book. Shared so every mode applies ops identically.
// Returns false only for a Cancel with nothing to cancel (no work done).
bool apply(Book& book, std::vector<OrderId>& live, const Op& op) {
    switch (op.kind) {
        case Op::Kind::Limit:
            live.push_back(book.limit(op.side, op.price, op.qty));
            return true;
        case Op::Kind::Market:
            book.market(op.side, op.qty);
            return true;
        case Op::Kind::Cancel: {
            if (live.empty()) return false;
            const OrderId id = live.back();
            live.pop_back();
            static_cast<void>(book.cancel(id));  // outcome is in the event log
            return true;
        }
        case Op::Kind::CancelId:
            static_cast<void>(book.cancel(op.id));  // outcome is in the event log
            return true;
    }
    return false;  // unreachable: the switch is exhaustive
}

// ---- generated workload ----------------------------------------------------

// Same LCG as the Rust bench, so `bench N` runs the same stream in both.
// Unsigned arithmetic wraps by definition, so this is well-defined C++.
class Lcg {
public:
    explicit Lcg(std::uint64_t seed) noexcept : s_(seed) {}
    std::uint64_t next() noexcept {
        s_ = s_ * 6364136223846793005ULL + 1442695040888963407ULL;
        return s_ >> 33;
    }
    std::int64_t range(std::int64_t lo, std::int64_t hi) noexcept {
        return lo + static_cast<std::int64_t>(next() % static_cast<std::uint64_t>(hi - lo + 1));
    }

private:
    std::uint64_t s_;
};

[[nodiscard]] std::vector<Op> make_stream(std::size_t n, std::uint64_t seed) {
    Lcg rng(seed);
    std::vector<Op> ops;
    ops.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        // Each draw gets its own statement: C++ leaves the evaluation order of
        // function arguments unspecified, and the stream must match Rust's.
        const Side side = rng.next() % 2 == 0 ? Side::Buy : Side::Sell;
        const std::int64_t roll = rng.range(0, 99);
        if (roll < 15) {
            ops.push_back(Op::cancel_last());
        } else if (roll < 25) {
            const Qty qty = rng.range(1, 100);
            ops.push_back(Op::market(side, qty));
        } else {
            const std::int64_t off = rng.range(1, 12);
            const Price price = side == Side::Buy ? 10'000 - off : 10'000 + off;
            const Qty qty = rng.range(1, 100);
            ops.push_back(Op::limit(side, price, qty));
        }
    }
    return ops;
}

Book run_stream(const std::vector<Op>& ops) {
    Book book;
    std::vector<OrderId> live;
    for (const Op& op : ops) apply(book, live, op);
    return book;
}

// ---- bench -----------------------------------------------------------------

[[nodiscard]] long long ns_between(SteadyClock::time_point a, SteadyClock::time_point b) {
    return static_cast<long long>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
}

// Median cost of reading the clock twice back to back: the floor under every
// per-op latency figure below.
[[nodiscard]] long long timer_overhead(std::size_t samples) {
    std::vector<long long> v;
    v.reserve(samples);
    for (std::size_t i = 0; i < samples; ++i) {
        const auto t = SteadyClock::now();
        v.push_back(ns_between(t, SteadyClock::now()));
    }
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2), v.end());
    return v[v.size() / 2];
}

int bench(std::optional<std::vector<Op>> ops_in, std::size_t n, std::size_t warmup) {
    std::printf("warmup %llu ops (discarded)\n", static_cast<unsigned long long>(warmup));
    static_cast<void>(run_stream(make_stream(warmup, 1)));

    using Period = SteadyClock::period;
    const long long oh = timer_overhead(20'000);
    std::printf("timer overhead (median): %lld ns\n", oh);
    std::printf("clock period (steady_clock): %.3g ns\n",
                1e9 * static_cast<double>(Period::num) / static_cast<double>(Period::den));
    if (oh <= 1) {
        std::printf("  NOTE: the clock barely resolves two back-to-back calls here.\n");
        std::printf("  Windows QueryPerformanceCounter ticks at roughly 100ns, so any\n");
        std::printf("  per-op figure near that is at the measurement floor. The\n");
        std::printf("  throughput number below does not use a per-op timer and is\n");
        std::printf("  therefore the more trustworthy one.\n");
    }
    std::printf("\n");

    const char* const source = ops_in ? "op file" : "generated";
    const std::vector<Op> ops = ops_in ? std::move(*ops_in) : make_stream(n, 42);
    n = ops.size();
    if (n == 0) {
        std::fprintf(stderr, "bench: no ops to run\n");
        return 1;
    }
    std::printf("workload: %s (%llu ops)\n\n", source, static_cast<unsigned long long>(n));

    // --- throughput: no per-op timing, so no timer cost in the number ---
    const auto t0 = SteadyClock::now();
    const Book book = run_stream(ops);
    const long long elapsed = ns_between(t0, SteadyClock::now());
    const double secs = static_cast<double>(elapsed) / 1e9;
    std::printf("THROUGHPUT  %llu ops in %.4fs\n", static_cast<unsigned long long>(n), secs);
    std::printf("            %.0f ops/sec  (%.0f ns/op)\n", static_cast<double>(n) / secs,
                static_cast<double>(elapsed) / static_cast<double>(n));
    std::printf("            %llu events emitted\n\n",
                static_cast<unsigned long long>(book.event_count()));

    // --- latency percentiles: each op timed individually ---
    Book timed;
    std::vector<OrderId> live;
    std::vector<long long> lat;
    lat.reserve(n);
    for (const Op& op : ops) {
        const auto t = SteadyClock::now();
        const bool ran = apply(timed, live, op);
        const auto done = SteadyClock::now();
        if (ran) lat.push_back(ns_between(t, done));
    }
    if (lat.empty()) {
        std::fprintf(stderr, "bench: no ops did any work\n");
        return 1;
    }
    std::sort(lat.begin(), lat.end());
    const auto pct = [&lat](double p) {
        const auto i = static_cast<std::size_t>(static_cast<double>(lat.size()) * p);
        return lat[std::min(i, lat.size() - 1)];
    };
    std::printf("LATENCY     per-op, timed individually (n=%llu)\n",
                static_cast<unsigned long long>(lat.size()));
    const std::pair<const char*, double> points[] = {
        {"p50", 0.50}, {"p90", 0.90}, {"p99", 0.99}, {"p99.9", 0.999}};
    for (const auto& [name, p] : points) {
        std::printf("            %5s: %9lld ns\n", name, pct(p));
    }
    std::printf("            %5s: %9lld ns\n\n", "max", lat.back());

    std::printf("note: single thread, -O3 build. Compare against the Python\n");
    std::printf("      baseline (bench.py --ops-file) on the SAME machine, or the\n");
    std::printf("      ratio is meaningless.\n");
    return 0;
}

// ---- replay ----------------------------------------------------------------

// Replay an op file and print the event log for diffing against Python/Rust.
// Parses and applies line by line, like the Python and Rust readers, so an
// engine rejection (say a zero price) is reported with its line number.
// Nothing is printed unless the whole file replays.
int replay(const std::string& path) {
    const std::string text = read_file(path);
    // Deterministic timestamps: this log is for comparison, so it must be
    // reproducible rather than reflecting whatever the clock said.
    Book book = Book::deterministic();
    std::vector<OrderId> live;
    // Rust's replay runs its full invariant check after every op. That check
    // is O(book), so over a long file it is quadratic (a 200,000-op replay
    // would take minutes, far longer under ASan). Compromise: the full check
    // after every one of the first 20,000 ops, which covers the standard diff
    // file exactly as Rust does, then every 1,000 ops and after the last; the
    // O(1) checks (crossed book, dense seq) still run after every op.
    constexpr std::size_t kFullCheckEveryOp = 20'000;
    constexpr std::size_t kFullCheckPeriod = 1'000;
    std::size_t applied = 0;
    for_each_line(text, [&](std::size_t line_no, std::string_view line) {
        try {
            if (const auto op = parse_line(line)) {
                apply(book, live, *op);
                ++applied;
                if (applied <= kFullCheckEveryOp || applied % kFullCheckPeriod == 0) {
                    book.check_invariants();
                } else {
                    book.check_touch();
                }
            }
        } catch (const std::exception& e) {
            rethrow_at(path, line_no, e);
        }
    });
    book.check_invariants();

    std::string out;
    out.reserve(book.event_count() * 24);
    for (const Event& e : book.events()) {
        append_encoded(out, e);
        out += '\n';
    }
#ifdef _WIN32
    // Text-mode stdout would turn every "\n" into "\r\n"; the log must be
    // byte-identical to the Rust binary's on every platform.
    static_cast<void>(_setmode(_fileno(stdout), _O_BINARY));
#endif
    if (std::fwrite(out.data(), 1, out.size(), stdout) != out.size() || std::fflush(stdout) != 0) {
        throw std::runtime_error("failed writing event log to stdout");
    }
    return 0;
}

// ---- selftest --------------------------------------------------------------

struct TestFailure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

void expect(bool cond, const char* what) {
    if (!cond) throw TestFailure(what);
}

template <typename F>
void expect_throws_invalid(F&& f, const char* what) {
    try {
        f();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw TestFailure(what);
}

[[nodiscard]] std::vector<ev::Trade> trades(const Book& b) {
    std::vector<ev::Trade> out;
    for (const Event& e : b.events()) {
        if (const auto* t = std::get_if<ev::Trade>(&e)) out.push_back(*t);
    }
    return out;
}

[[nodiscard]] std::vector<std::string> log_of(const Book& b) {
    std::vector<std::string> out;
    for (const Event& e : b.events()) out.push_back(encode(e));
    return out;
}

using Depth = std::vector<std::pair<Price, Qty>>;

void test_exact_fill_and_encoding() {
    Book b = Book::deterministic();
    b.limit(Side::Sell, 100, 10);
    b.limit(Side::Buy, 100, 10);
    b.market(Side::Buy, 3);
    const std::vector<std::string> want = {"1 A 1 S 100 10", "2 A 2 B 100 10", "3 T 100 10 2 1",
                                           "4 A 3 B M 3", "5 C 3 3"};
    expect(log_of(b) == want, "encoded log differs from the diff_test wire format");
    expect(!b.best_bid() && !b.best_ask(), "book should be empty");
    b.check_invariants();
}

void test_partial_fill_rests_remainder() {
    Book b;
    const OrderId ask = b.limit(Side::Sell, 100, 5);
    const OrderId bid = b.limit(Side::Buy, 100, 8);
    expect(b.best_bid() == Price{100}, "remainder should rest as the best bid");
    expect(b.depth(Side::Buy) == Depth{{100, 3}}, "resting remainder should be 3");
    expect(b.has_order(bid) && !b.has_order(ask), "filled order must leave the index");
    b.check_invariants();
}

void test_fifo_within_level() {
    Book b;
    const OrderId first = b.limit(Side::Sell, 100, 5);
    const OrderId second = b.limit(Side::Sell, 100, 5);
    b.limit(Side::Buy, 100, 5);
    const auto t = trades(b);
    expect(t.size() == 1 && t[0].resting == first, "FIFO violated");
    expect(b.has_order(second) && b.depth(Side::Sell) == Depth{{100, 5}},
           "second order should be untouched");
    b.check_invariants();
}

void test_better_price_beats_earlier_time() {
    Book b;
    b.limit(Side::Sell, 101, 5);
    const OrderId cheap = b.limit(Side::Sell, 100, 5);
    b.limit(Side::Buy, 101, 5);
    const auto t = trades(b);
    expect(t.size() == 1 && t[0].resting == cheap, "price priority violated");
    b.check_invariants();
}

void test_fills_at_resting_price() {
    Book b;
    b.limit(Side::Sell, 100, 5);
    b.limit(Side::Buy, 105, 5);
    expect(trades(b).at(0).price == 100, "must fill at the resting order's price");
    b.check_invariants();
}

void test_market_never_rests() {
    Book b;
    b.limit(Side::Sell, 100, 3);
    b.limit(Side::Sell, 101, 3);
    const OrderId m = b.market(Side::Buy, 10);
    const auto t = trades(b);
    expect(t.size() == 2 && t[0].qty == 3 && t[1].qty == 3, "market should sweep both levels");
    const auto* c = std::get_if<ev::Canceled>(&b.events().back());
    expect(c && c->id == m && c->remaining == 4, "unfilled remainder must be canceled");
    expect(!b.best_bid() && !b.best_ask() && !b.has_order(m), "market order must not rest");

    // Into an empty book: accepted, then canceled in full, never rests.
    const OrderId e = b.market(Side::Sell, 7);
    const auto* c2 = std::get_if<ev::Canceled>(&b.events().back());
    expect(c2 && c2->id == e && c2->remaining == 7 && !b.has_order(e),
           "market into empty book must cancel in full");
    b.check_invariants();
}

void test_cancel_then_double_cancel() {
    Book b;
    const OrderId oid = b.limit(Side::Sell, 100, 10);
    b.limit(Side::Buy, 100, 4);
    expect(b.cancel(oid), "cancel of a live order must succeed");
    const auto* c = std::get_if<ev::Canceled>(&b.events().back());
    expect(c && c->id == oid && c->remaining == 6, "cancel reports the unfilled 6");
    expect(!b.best_ask(), "cancelled level must disappear");
    expect(!b.cancel(oid), "double cancel must fail");
    const auto* r = std::get_if<ev::RejectCancel>(&b.events().back());
    expect(r && r->id == oid, "double cancel must emit reject_cancel");
    b.check_invariants();
}

void test_fully_cancelled_level_stops_advertising() {
    Book b;
    const OrderId a = b.limit(Side::Buy, 100, 5);
    const OrderId c = b.limit(Side::Buy, 100, 5);
    expect(b.cancel(a), "cancel a");
    expect(b.best_bid() == Price{100}, "level still has a live order");
    expect(b.depth(Side::Buy) == Depth{{100, 5}}, "depth must skip the husk");
    expect(b.cancel(c), "cancel c");
    expect(!b.best_bid(), "phantom level: all orders cancelled");
    b.limit(Side::Buy, 100, 7);  // re-rest at a price whose level was erased
    expect(b.depth(Side::Buy) == Depth{{100, 7}}, "re-rested level should show 7");
    b.check_invariants();
}

void test_cancelled_husk_never_filled() {
    Book b;
    const OrderId dead = b.limit(Side::Sell, 100, 5);
    const OrderId alive = b.limit(Side::Sell, 100, 5);
    expect(b.cancel(dead), "cancel dead");
    b.limit(Side::Buy, 100, 5);
    const auto t = trades(b);
    expect(t.size() == 1 && t[0].resting == alive, "filled a cancelled order");
    b.check_invariants();
}

void test_cancel_never_issued_ids() {
    Book b = Book::deterministic();
    expect(!b.cancel(0), "id 0 is never issued");
    expect(!b.cancel(1), "id 1 not issued yet");
    expect(!b.cancel(std::numeric_limits<OrderId>::max()), "max id never issued");
    const std::vector<std::string> want = {"1 R 0", "2 R 1", "3 R 18446744073709551615"};
    expect(log_of(b) == want, "rejects must be logged with dense seq");

    const OrderId m = b.market(Side::Buy, 5);  // issued, but never rested
    expect(!b.cancel(m), "a market order id is never cancellable");
    const OrderId s = b.limit(Side::Sell, 100, 5);
    b.limit(Side::Buy, 100, 5);
    expect(!b.cancel(s), "a filled order is no longer cancellable");
    expect(!b.best_bid() && !b.best_ask(), "rejected cancels must change nothing");
    b.check_invariants();
}

void test_extreme_prices() {
    // 2^53 is where doubles stop representing every integer: 2^53 + 1 rounds
    // to 2^53 as a double. An engine that let a float near its prices would
    // merge these two levels; integer ticks keep them apart.
    constexpr Price p53 = Price{1} << 53;
    Book b = Book::deterministic();
    const OrderId hi = b.limit(Side::Sell, p53 + 1, 2);
    const OrderId mid = b.limit(Side::Sell, p53, 3);
    const OrderId lo = b.limit(Side::Sell, 1, 1);
    expect(b.depth(Side::Sell) == Depth{{1, 1}, {p53, 3}, {p53 + 1, 2}},
           "2^53 and 2^53+1 must be distinct levels");
    const OrderId buy = b.limit(Side::Buy, p53 + 1, 10);
    const auto t = trades(b);
    expect(t.size() == 3 && t[0].resting == lo && t[0].price == 1 && t[1].resting == mid &&
               t[1].price == p53 && t[2].resting == hi && t[2].price == p53 + 1,
           "sweep must go 1, 2^53, 2^53+1 at resting prices");
    const auto log = log_of(b);
    expect(log.at(5) == "6 T 9007199254740992 3 4 2" && log.at(6) == "7 T 9007199254740993 2 4 1",
           "extreme prices must encode exactly");
    expect(b.best_bid() == p53 + 1 && b.depth(Side::Buy) == Depth{{p53 + 1, 4}} &&
               b.has_order(buy),
           "remainder 4 rests at 2^53+1");
    b.check_invariants();

    Book m;  // the full int64 range is legal, as in Python
    m.limit(Side::Buy, std::numeric_limits<Price>::max(), 1);
    m.limit(Side::Sell, 1, 1);
    expect(trades(m).at(0).price == std::numeric_limits<Price>::max(), "int64 max price");
    m.check_invariants();
}

void test_invalid_input_rejected() {
    Book b;
    expect_throws_invalid([&] { b.limit(Side::Buy, 0, 5); }, "price 0 accepted");
    expect_throws_invalid([&] { b.limit(Side::Buy, -1, 5); }, "negative price accepted");
    expect_throws_invalid([&] { b.limit(Side::Buy, 100, 0); }, "qty 0 accepted");
    expect_throws_invalid([&] { b.limit(Side::Sell, 100, -5); }, "negative qty accepted");
    expect_throws_invalid([&] { b.market(Side::Buy, 0); }, "market qty 0 accepted");
    expect_throws_invalid([&] { b.limit(static_cast<Side>('X'), 100, 5); }, "bad side accepted");
    // b.limit(Side::Buy, 100.5, 5) does not compile: see the deleted overloads.
    expect(b.event_count() == 0 && !b.best_bid(), "rejected orders left residue");
    expect(b.limit(Side::Buy, 100, 5) == 1, "rejected orders must not consume ids");
    b.check_invariants();
}

void test_seq_and_clocks() {
    Book b;  // steady clock
    b.limit(Side::Sell, 100, 5);
    b.limit(Side::Buy, 100, 5);
    const OrderId oid = b.limit(Side::Buy, 99, 5);
    expect(b.cancel(oid), "cancel");
    expect(!b.cancel(9999), "reject");
    std::uint64_t prev_ts = 0;
    for (std::size_t i = 0; i < b.events().size(); ++i) {
        expect(seq_of(b.events()[i]) == i + 1, "seq must be dense and monotonic");
        expect(ts_of(b.events()[i]) >= prev_ts, "timestamps went backwards");
        prev_ts = ts_of(b.events()[i]);
    }
    const auto run = [] {
        Book d = Book::deterministic();
        d.limit(Side::Sell, 100, 5);
        d.limit(Side::Buy, 100, 5);
        return d;
    };
    const Book d1 = run();
    const Book d2 = run();
    expect(log_of(d1) == log_of(d2), "deterministic clock must reproduce");
    for (const Event& e : d1.events()) expect(ts_of(e) == seq_of(e), "deterministic ts == seq");
}

void test_portable_accessors() {
    Book b;
    const OrderId a = b.limit(Side::Buy, 99, 5);
    const OrderId c = b.limit(Side::Buy, 99, 7);
    b.limit(Side::Buy, 98, 1);
    b.limit(Side::Sell, 101, 2);
    expect(b.cancel(a), "cancel");
    expect(b.depth(Side::Buy, 5) == Depth{{99, 7}, {98, 1}}, "bid depth");
    expect(b.depth(Side::Buy, 1) == Depth{{99, 7}}, "depth respects level cap");
    expect(b.depth(Side::Sell, 5) == Depth{{101, 2}}, "ask depth");
    expect(!b.has_order(a) && b.has_order(c), "has_order");
    expect(b.order_ids() == std::vector<OrderId>{2, 3, 4}, "order_ids ascending");
    b.check_invariants();
}

void test_randomized_stream_conserves_shares() {
    Lcg rng(42);
    Book b;
    Qty submitted = 0;
    std::vector<OrderId> live_ids;
    for (int i = 0; i < 20'000; ++i) {
        const Side side = rng.next() % 2 == 0 ? Side::Buy : Side::Sell;
        const std::int64_t roll = rng.range(0, 99);
        if (roll < 15 && !live_ids.empty()) {
            const std::size_t k = static_cast<std::size_t>(rng.next() % live_ids.size());
            std::swap(live_ids[k], live_ids.back());
            const OrderId id = live_ids.back();
            live_ids.pop_back();
            static_cast<void>(b.cancel(id));  // may already have filled; both are fine
        } else if (roll < 30) {
            const Qty q = rng.range(1, 20);
            submitted += q;
            b.market(side, q);
        } else {
            const Qty q = rng.range(1, 20);
            const Price p = rng.range(95, 105);
            submitted += q;
            live_ids.push_back(b.limit(side, p, q));
        }
        b.check_invariants();
    }
    Qty filled = 0;
    Qty canceled = 0;
    for (const Event& e : b.events()) {
        if (const auto* t = std::get_if<ev::Trade>(&e)) filled += t->qty;
        if (const auto* c = std::get_if<ev::Canceled>(&e)) canceled += c->remaining;
    }
    // Each fill consumes `qty` from BOTH the aggressor and the resting order,
    // so a trade of size q accounts for 2q of the submitted shares.
    expect(2 * filled + canceled + b.resting_qty() == submitted, "shares leaked");
    expect(filled > 0 && canceled > 0 && b.resting_qty() > 0, "stream should exercise all paths");
}

void test_parser() {
    const auto ok = parse_line("  L S 10004 18\r");
    expect(ok && ok->kind == Op::Kind::Limit && ok->side == Side::Sell && ok->price == 10004 &&
               ok->qty == 18,
           "valid limit line");
    const auto x = parse_line("X 18446744073709551615");
    expect(x && x->kind == Op::Kind::CancelId && x->id == std::numeric_limits<OrderId>::max(),
           "max u64 id");
    expect(!parse_line("") && !parse_line("   \r") && !parse_line("# comment"), "skip lines");
    for (const char* bad : {"L B 100.5 10", "L B 100 1.0", "L B 100.0 5", "L B 1e3 5",
                            "L B +100 5", "L Q 100 5", "L B 100", "L B 100 5 9", "Z 1", "X -1",
                            "X 1e3", "M S 99999999999999999999", "L B 9223372036854775808 1"}) {
        expect_throws_invalid([bad] { static_cast<void>(parse_line(bad)); }, bad);
    }
}

int selftest() {
    const std::pair<const char*, void (*)()> tests[] = {
        {"exact fill and wire encoding", test_exact_fill_and_encoding},
        {"partial fill rests remainder", test_partial_fill_rests_remainder},
        {"FIFO within a price level", test_fifo_within_level},
        {"better price beats earlier time", test_better_price_beats_earlier_time},
        {"fills at the resting price", test_fills_at_resting_price},
        {"market order never rests", test_market_never_rests},
        {"cancel then double cancel", test_cancel_then_double_cancel},
        {"fully cancelled level stops advertising", test_fully_cancelled_level_stops_advertising},
        {"cancelled husk is never filled", test_cancelled_husk_never_filled},
        {"cancels of never-issued ids", test_cancel_never_issued_ids},
        {"extreme prices 1, 2^53, int64 max", test_extreme_prices},
        {"invalid input rejected without residue", test_invalid_input_rejected},
        {"dense seq, monotonic and deterministic clocks", test_seq_and_clocks},
        {"portable accessors", test_portable_accessors},
        {"randomized stream conserves shares", test_randomized_stream_conserves_shares},
        {"op-file parser rejects malformed lines", test_parser},
    };
    int failed = 0;
    for (const auto& [name, fn] : tests) {
        try {
            fn();
            std::printf("ok    %s\n", name);
        } catch (const std::exception& e) {
            ++failed;
            std::printf("FAIL  %s: %s\n", name, e.what());
        }
    }
    const int total = static_cast<int>(std::size(tests));
    std::printf("\nselftest: %d passed, %d failed\n", total - failed, failed);
    return failed == 0 ? 0 : 1;
}

int usage() {
    std::fprintf(stderr,
                 "usage: bullpen_cpp replay <opsfile>\n"
                 "       bullpen_cpp bench [opsfile|N]\n"
                 "       bullpen_cpp selftest\n");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    const std::vector<std::string> args(argv, argv + argc);
    const std::string mode = args.size() > 1 ? args[1] : "bench";
    try {
        if (mode == "replay") {
            if (args.size() != 3) return usage();
            return replay(args[2]);
        }
        if (mode == "bench") {
            if (args.size() > 3) return usage();
            if (args.size() == 2) return bench(std::nullopt, 200'000, 20'000);
            // `bench N` generates a stream; `bench <file>` replays a shared op
            // file so every engine is measured on identical work.
            std::size_t n = 0;
            const std::string& a = args[2];
            const auto res = std::from_chars(a.data(), a.data() + a.size(), n);
            if (res.ec == std::errc{} && res.ptr == a.data() + a.size()) {
                return bench(std::nullopt, n, 20'000);
            }
            return bench(load_ops(a), 0, 20'000);
        }
        if (mode == "selftest") return selftest();
        return usage();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "bullpen_cpp %s: %s\n", mode.c_str(), e.what());
        return 1;
    }
}
