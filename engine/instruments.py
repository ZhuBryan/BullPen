"""
instruments.py — reference data and multi-symbol routing.

An engine that trades one nameless instrument has skipped the layer every
venue has: the instrument master. Before an order can be matched, somebody has
to answer what the minimum price increment is, what a valid lot looks like,
and whether this symbol is even listed here.

TICK SIZE IS NOT CONSTANT. US equities follow SEC Reg NMS Rule 612, the
sub-penny rule: quotes at or above $1.00 must be in $0.01 increments, below
$1.00 in $0.0001. So a $0.50 stock can be quoted at $0.5012 and a $50 stock
cannot be quoted at $50.001. Hardcoding a single tick size means either
rejecting legitimate sub-dollar prices or accepting sub-penny quotes that a
real venue would bounce.

PRICES BECOME INTEGER TICKS at the boundary. The matching engine works in
integers because float money is silent death (see orderbook.py), so this module
owns the conversion, using the smallest increment in the regime as the base
unit. Converting with int(price / base) truncates 0.29999999 to the wrong tick,
so the rounding here is deliberate.

Run `python instruments.py` for the self-check.
"""

import math
from dataclasses import dataclass, field

try:
    from engine.orderbook import BUY, SELL, Book
except ImportError:
    from orderbook import BUY, SELL, Book


class InstrumentError(ValueError):
    """Rejected before the order ever reaches the book."""


@dataclass(frozen=True)
class TickRule:
    """At or above `threshold`, the increment is `tick`."""
    threshold: float
    tick: float


# SEC Reg NMS Rule 612. Order matters: evaluated highest threshold first.
US_EQUITY_TICKS = (TickRule(1.00, 0.01), TickRule(0.00, 0.0001))
# Crypto venues quote far finer and do not have a sub-dollar regime change.
CRYPTO_TICKS = (TickRule(0.00, 0.01),)


@dataclass
class Instrument:
    symbol: str
    tick_rules: tuple = US_EQUITY_TICKS
    lot_size: int = 1
    min_qty: int = 1
    max_qty: int = 1_000_000
    multiplier: int = 1          # contract size; 100 for an equity option
    currency: str = "USD"
    tradable: bool = True

    @property
    def base_increment(self) -> float:
        """Smallest increment in the regime: the integer tick unit."""
        return min(r.tick for r in self.tick_rules)

    def tick_size(self, price: float) -> float:
        for rule in sorted(self.tick_rules, key=lambda r: -r.threshold):
            if price >= rule.threshold:
                return rule.tick
        return self.tick_rules[-1].tick

    # ---- price <-> integer ticks ---------------------------------------
    def to_ticks(self, price: float) -> int:
        """Convert a price to the integer the matching engine works in."""
        if price <= 0:
            raise InstrumentError(f"price must be positive, got {price}")
        # round(), not int(): truncation sends 0.29999999999 to the tick below.
        return int(round(price / self.base_increment))

    def from_ticks(self, ticks: int) -> float:
        return round(ticks * self.base_increment, 10)

    def round_price(self, price: float, side=None) -> float:
        """Snap to a valid increment.

        With a side, round conservatively: a buyer rounds DOWN and a seller
        rounds UP, so snapping never makes an order more aggressive than the
        trader asked for.
        """
        tick = self.tick_size(price)
        n = price / tick
        # The epsilon matters: 185.25 / 0.01 is 18525.000000000004, so a bare
        # ceil() would push an already-valid price up a whole tick.
        if side == BUY:
            n = math.floor(n + 1e-9)
        elif side == SELL:
            n = math.ceil(n - 1e-9)
        else:
            n = round(n)
        return round(n * tick, 10)

    # ---- validation ----------------------------------------------------
    def validate_price(self, price: float):
        if not self.tradable:
            raise InstrumentError(f"{self.symbol} is not tradable")
        if price is None:
            return                      # market order
        if price <= 0:
            raise InstrumentError(f"price must be positive, got {price}")
        tick = self.tick_size(price)
        # Compare in tick units to avoid float modulo noise.
        if abs(round(price / tick) - price / tick) > 1e-6:
            raise InstrumentError(
                f"{price} violates the {tick} increment for {self.symbol}")

    def validate_qty(self, qty: int):
        if qty != int(qty):
            raise InstrumentError(f"quantity must be whole, got {qty}")
        qty = int(qty)
        if qty < self.min_qty:
            raise InstrumentError(f"quantity {qty} below minimum {self.min_qty}")
        if qty > self.max_qty:
            raise InstrumentError(f"quantity {qty} above maximum {self.max_qty}")
        if qty % self.lot_size:
            raise InstrumentError(
                f"quantity {qty} is not a multiple of lot size {self.lot_size}")

    def validate(self, price, qty):
        self.validate_price(price)
        self.validate_qty(qty)


class Venue:
    """A set of listed instruments, each with its own book."""

    def __init__(self, instruments=None):
        self.instruments: dict[str, Instrument] = {}
        self.books: dict[str, Book] = {}
        for ins in instruments or []:
            self.list(ins)

    def list(self, instrument: Instrument):
        if instrument.symbol in self.instruments:
            raise InstrumentError(f"{instrument.symbol} is already listed")
        self.instruments[instrument.symbol] = instrument
        self.books[instrument.symbol] = Book()
        return instrument

    def instrument(self, symbol) -> Instrument:
        ins = self.instruments.get(symbol)
        if ins is None:
            raise InstrumentError(f"unknown symbol {symbol}")
        return ins

    def book(self, symbol) -> Book:
        self.instrument(symbol)          # raises on unknown symbol
        return self.books[symbol]

    def submit(self, symbol, side, qty, price=None):
        """Validate against reference data, then route to the right book.

        Prices are converted to integer ticks here, which is the only place
        that conversion should happen.
        """
        ins = self.instrument(symbol)
        ins.validate(price, qty)
        book = self.books[symbol]
        if price is None:
            return book.market(side, int(qty))
        return book.limit(side, ins.to_ticks(price), int(qty))

    def best(self, symbol):
        """Top of book in real prices rather than ticks."""
        ins, book = self.instrument(symbol), self.books[symbol]
        bid, ask = book.best_bid(), book.best_ask()
        return (ins.from_ticks(bid) if bid is not None else None,
                ins.from_ticks(ask) if ask is not None else None)

    def snapshot(self):
        out = {}
        for sym in self.instruments:
            bid, ask = self.best(sym)
            out[sym] = {"bid": bid, "ask": ask,
                        "spread": round(ask - bid, 10) if bid and ask else None}
        return out


def demo():
    eq = Instrument("AAPL")

    # ---- Reg NMS 612: the increment changes at $1.00 ----
    assert eq.tick_size(185.25) == 0.01
    assert eq.tick_size(1.00) == 0.01, "the rule applies AT one dollar"
    assert eq.tick_size(0.9999) == 0.0001
    assert eq.tick_size(0.50) == 0.0001

    eq.validate_price(185.25)
    eq.validate_price(0.5012)                 # legal sub-dollar quote
    for bad in (185.255, 1.005, 50.001):
        try:
            eq.validate_price(bad)
            raise AssertionError(f"accepted sub-penny {bad}")
        except InstrumentError:
            pass
    for bad in (0, -1):
        try:
            eq.validate_price(bad); raise AssertionError("accepted non-positive")
        except InstrumentError:
            pass

    # ---- integer tick conversion survives float representation ----
    assert eq.to_ticks(185.25) == 1_852_500
    assert eq.from_ticks(eq.to_ticks(185.25)) == 185.25
    for p in (0.0001, 0.3, 1.00, 29.99, 185.25, 999.99):
        assert eq.from_ticks(eq.to_ticks(p)) == p, p
    # the case that truncation gets wrong
    assert eq.to_ticks(0.29999999999) == 3000

    # ---- conservative rounding never increases aggression ----
    assert eq.round_price(185.256) == 185.26          # nearest
    assert eq.round_price(185.256, side=BUY) == 185.25, "a buyer rounds down"
    assert eq.round_price(185.251, side=SELL) == 185.26, "a seller rounds up"
    assert eq.round_price(185.25, side=BUY) == 185.25, "already valid, unchanged"
    assert eq.round_price(185.25, side=SELL) == 185.25

    # ---- lots and size bounds ----
    lot = Instrument("LOT100", lot_size=100, min_qty=100, max_qty=10_000)
    lot.validate_qty(500)
    for bad in (50, 150, 20_000, 1.5):
        try:
            lot.validate_qty(bad); raise AssertionError(f"accepted qty {bad}")
        except InstrumentError:
            pass

    halted = Instrument("HALT", tradable=False)
    try:
        halted.validate_price(10.00); raise AssertionError("traded a halted name")
    except InstrumentError:
        pass

    # ---- multi-symbol routing: books stay independent ----
    v = Venue([Instrument("AAPL"), Instrument("MSFT"),
               Instrument("BTCUSD", tick_rules=CRYPTO_TICKS)])
    v.submit("AAPL", SELL, 100, 185.25)
    v.submit("AAPL", BUY, 60, 185.25)
    v.submit("MSFT", BUY, 10, 400.00)

    assert v.best("AAPL") == (None, 185.25), v.best("AAPL")
    assert v.best("MSFT") == (400.00, None)
    assert v.book("AAPL") is not v.book("MSFT"), "books must not be shared"
    trades = [e for e in v.book("AAPL").events if e[0] == "trade"]
    assert len(trades) == 1 and trades[0][2] == 60
    assert not [e for e in v.book("MSFT").events if e[0] == "trade"], \
        "a trade in one symbol must not appear in another"

    snap = v.snapshot()
    assert set(snap) == {"AAPL", "MSFT", "BTCUSD"}
    assert snap["BTCUSD"]["bid"] is None and snap["BTCUSD"]["spread"] is None

    # crypto has no sub-dollar regime change
    btc = v.instrument("BTCUSD")
    assert btc.tick_size(0.5) == 0.01 and btc.tick_size(68_000) == 0.01

    # unknown symbols and invalid orders are refused before reaching a book
    for call in (lambda: v.submit("NOPE", BUY, 1, 1.00),
                 lambda: v.submit("AAPL", BUY, 1, 185.2555),
                 lambda: v.book("NOPE")):
        try:
            call(); raise AssertionError("accepted an invalid instruction")
        except InstrumentError:
            pass
    try:
        v.list(Instrument("AAPL")); raise AssertionError("double listing accepted")
    except InstrumentError:
        pass

    # a market order needs no price but still obeys size rules
    v.submit("MSFT", SELL, 10, None)
    assert v.best("MSFT") == (None, None), "market order should have crossed"

    print("ok — Reg NMS tick regimes, tick conversion, lot rules, multi-symbol routing")


if __name__ == "__main__":
    demo()
