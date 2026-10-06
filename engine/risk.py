"""
risk.py — pre-trade risk checks, the gate in front of the matching engine.

Real venues do not let an order reach the book unchecked. A gateway or OMS
applies limits first, and the matching engine is then free to assume every
order it sees is already legal. That separation is deliberate: risk logic
changes often and per-account, matching logic must not.

The five checks here are the ones that actually exist in production:

  MAX ORDER SIZE / VALUE  Catches the fat-finger order. A trader meaning to buy
      100 types 100000, and without a cap that order reaches the book.

  PRICE COLLAR  Rejects a limit far away from the current market. A buy at 10x
      the last trade is almost never intentional, and if it rests it will be
      picked off the moment the market moves. Measured in basis points against
      a reference price, so it scales with the instrument.

  POSITION LIMIT  Caps net exposure per account, enforced against fills rather
      than submissions, because an order that never fills creates no risk.

  OPEN ORDER LIMIT  Caps resting orders per account. Partly risk, partly a
      cheap defence against a runaway algo flooding the book.

  SELF-TRADE PREVENTION  Stops an account matching its own resting order.
      Those trades move no real risk but do print to the tape, which is why
      regulators treat them as potential wash trading. Here the aggressor is
      rejected; real venues also offer cancel-newest, cancel-oldest, and
      decrement-and-cancel, which differ in who loses queue position.

Run `python risk.py` for the self-check.
"""

from dataclasses import dataclass

try:
    from engine.orderbook import BUY, SELL, Book
except ImportError:
    from orderbook import BUY, SELL, Book


@dataclass
class RiskLimits:
    max_order_qty: int = 10_000
    max_order_value: int = 100_000_000   # price_ticks * qty
    collar_bps: float = 1_000.0          # 10% away from reference
    max_position: int = 50_000           # absolute net position
    max_open_orders: int = 500
    allow_self_trade: bool = False


class RiskReject(Exception):
    """Carries a machine-readable reason, the way a venue's reject does."""

    def __init__(self, code, detail=""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


class RiskGate:
    """Wraps a Book. Same submit API, plus an account and the checks."""

    def __init__(self, book: Book = None, limits: RiskLimits = None):
        self.book = book or Book()
        self.limits = limits or RiskLimits()
        self.owner: dict[int, str] = {}          # order id -> account
        # Side is recorded at submission. Inferring it from the book fails
        # exactly when a fill consumes BOTH orders, since neither is left in
        # book.orders to ask.
        self.side_of: dict[int, str] = {}
        self.position: dict[str, int] = {}       # account -> net shares
        self.open_orders: dict[str, int] = {}    # account -> resting count
        self.rejects: list[tuple] = []
        self._seen = 0                           # events consumed so far

    # ---- introspection -------------------------------------------------
    def reference_price(self):
        """Last traded price, else the mid, else None (no collar possible)."""
        for ev in reversed(self.book.events):
            if ev[0] == "trade":
                return ev[1]
        bb, ba = self.book.best_bid(), self.book.best_ask()
        if bb is not None and ba is not None:
            return (bb + ba) // 2
        return None

    # ---- the checks ----------------------------------------------------
    def _check(self, account, side, price, qty):
        lim = self.limits
        if qty <= 0:
            raise RiskReject("INVALID_QTY", str(qty))
        if qty > lim.max_order_qty:
            raise RiskReject("MAX_ORDER_QTY", f"{qty} > {lim.max_order_qty}")

        if price is not None:
            if price <= 0:
                raise RiskReject("INVALID_PRICE", str(price))
            if price * qty > lim.max_order_value:
                raise RiskReject("MAX_ORDER_VALUE", f"{price * qty} > {lim.max_order_value}")

            ref = self.reference_price()
            if ref:
                away_bps = abs(price - ref) / ref * 10_000
                if away_bps > lim.collar_bps:
                    raise RiskReject(
                        "PRICE_COLLAR",
                        f"{away_bps:.0f}bps from {ref} exceeds {lim.collar_bps:.0f}bps")

        if self.open_orders.get(account, 0) >= lim.max_open_orders:
            raise RiskReject("MAX_OPEN_ORDERS", str(lim.max_open_orders))

        # Worst case assumes the whole order fills, which is the only safe way
        # to check a limit before knowing what it trades against.
        signed = qty if side == BUY else -qty
        projected = self.position.get(account, 0) + signed
        if abs(projected) > lim.max_position:
            raise RiskReject("MAX_POSITION", f"{projected} exceeds {lim.max_position}")

        if not lim.allow_self_trade:
            self._check_self_trade(account, side, price, qty)

    def _check_self_trade(self, account, side, price, qty):
        """Would this order cross against a resting order of the same account?"""
        levels, prices, _ = self.book._levels(SELL if side == BUY else BUY)
        for p in (prices if side == BUY else reversed(prices)):
            if price is not None:
                crosses = p <= price if side == BUY else p >= price
                if not crosses:
                    break
            for o in levels[p]:
                if not o.cancelled and self.owner.get(o.id) == account:
                    raise RiskReject("SELF_TRADE", f"own order {o.id} at {p}")

    # ---- submission ----------------------------------------------------
    def limit(self, account, side, price, qty):
        return self._submit(account, side, int(price), qty)

    def market(self, account, side, qty):
        return self._submit(account, side, None, qty)

    def _submit(self, account, side, price, qty):
        try:
            self._check(account, side, price, qty)
        except RiskReject as e:
            self.rejects.append((account, e.code, e.detail))
            raise

        oid = (self.book.market(side, qty) if price is None
               else self.book.limit(side, price, qty))
        self.owner[oid] = account
        self.side_of[oid] = side
        if oid in self.book.orders:           # it rested
            self.open_orders[account] = self.open_orders.get(account, 0) + 1
        self._settle()
        return oid

    def cancel(self, account, order_id):
        if self.owner.get(order_id) != account:
            self.rejects.append((account, "NOT_YOUR_ORDER", str(order_id)))
            raise RiskReject("NOT_YOUR_ORDER", str(order_id))
        resting = order_id in self.book.orders
        ok = self.book.cancel(order_id)
        if ok and resting:
            self.open_orders[account] = max(0, self.open_orders.get(account, 0) - 1)
        return ok

    def _settle(self):
        """Apply new fills to positions and open-order counts."""
        for ev in self.book.events[self._seen:]:
            if ev[0] != "trade":
                continue
            _, _, qty, aggressor, resting = ev[:5]
            for oid in (aggressor, resting):
                acct = self.owner.get(oid)
                side = self.side_of.get(oid)
                if acct is None or side is None:
                    continue          # an order this gate did not submit
                self.position[acct] = (self.position.get(acct, 0)
                                       + (qty if side == BUY else -qty))
            # a resting order that filled completely is no longer open
            if resting not in self.book.orders:
                acct = self.owner.get(resting)
                if acct:
                    self.open_orders[acct] = max(0, self.open_orders.get(acct, 0) - 1)
        self._seen = len(self.book.events)


def demo():
    def gate(**kw):
        return RiskGate(Book(), RiskLimits(**kw))

    def rejected(fn, code):
        try:
            fn()
        except RiskReject as e:
            assert e.code == code, f"expected {code}, got {e.code}"
            return
        raise AssertionError(f"expected a {code} reject")

    # ---- size and value caps ----
    g = gate(max_order_qty=100)
    rejected(lambda: g.limit("A", BUY, 100, 101), "MAX_ORDER_QTY")
    rejected(lambda: g.limit("A", BUY, 100, 0), "INVALID_QTY")
    assert g.limit("A", BUY, 100, 100), "a legal order must pass"

    g = gate(max_order_value=1_000)
    rejected(lambda: g.limit("A", BUY, 100, 11), "MAX_ORDER_VALUE")
    assert g.limit("A", BUY, 100, 10)

    # ---- price collar, measured against the last trade ----
    g = gate(collar_bps=500)                      # 5%
    g.limit("A", SELL, 10_000, 10)
    g.limit("B", BUY, 10_000, 10)                 # trade prints at 10,000
    assert g.reference_price() == 10_000
    rejected(lambda: g.limit("A", BUY, 20_000, 1), "PRICE_COLLAR")
    rejected(lambda: g.limit("A", BUY, 5_000, 1), "PRICE_COLLAR")
    assert g.limit("A", BUY, 9_800, 1), "2% away should pass a 5% collar"

    # with no trades and a one-sided book there is no reference, so no collar
    g = gate(collar_bps=1)
    assert g.limit("A", BUY, 50_000, 1), "collar cannot apply without a reference"

    # ---- position limit, enforced on fills not submissions ----
    g = gate(max_position=100)
    rejected(lambda: g.limit("A", BUY, 100, 101), "MAX_POSITION")
    g.limit("M1", SELL, 100, 60)
    g.limit("A", BUY, 100, 60)                    # A fills 60 long
    assert g.position["A"] == 60, g.position
    assert g.position["M1"] == -60, g.position
    # A fresh maker: the limit is per account, and M1 is already at -60, so
    # selling another 60 would breach its own limit too.
    rejected(lambda: g.limit("M1", SELL, 100, 60), "MAX_POSITION")
    g.limit("M2", SELL, 100, 60)
    rejected(lambda: g.limit("A", BUY, 100, 60), "MAX_POSITION")   # 60+60 > 100
    assert g.limit("A", BUY, 100, 40), "40 more keeps A at the 100 limit"
    assert g.position["A"] == 100, g.position

    # ---- open order cap, and cancelling frees a slot ----
    g = gate(max_open_orders=2)
    a = g.limit("A", BUY, 99, 1)
    g.limit("A", BUY, 98, 1)
    rejected(lambda: g.limit("A", BUY, 97, 1), "MAX_OPEN_ORDERS")
    g.cancel("A", a)
    assert g.limit("A", BUY, 97, 1), "cancelling should free a slot"

    # cancelling someone else's order is refused
    g = gate()
    oid = g.limit("A", BUY, 99, 1)
    rejected(lambda: g.cancel("B", oid), "NOT_YOUR_ORDER")
    assert g.cancel("A", oid) is True

    # ---- self-trade prevention ----
    g = gate()
    g.limit("A", SELL, 100, 10)
    rejected(lambda: g.limit("A", BUY, 100, 5), "SELF_TRADE")
    rejected(lambda: g.market("A", BUY, 5), "SELF_TRADE")   # market crosses too
    assert g.limit("B", BUY, 100, 5), "another account may cross it"
    # a non-crossing order from the same account is fine
    assert g.limit("A", BUY, 99, 5), "resting below your own ask is not a self-trade"

    # explicitly allowed when configured
    g = gate(allow_self_trade=True)
    g.limit("A", SELL, 100, 10)
    assert g.limit("A", BUY, 100, 5), "self-trade should pass when permitted"

    # ---- the gate must not change matching behaviour ----
    plain = Book()
    plain.limit(SELL, 100, 10)
    plain.limit(BUY, 100, 4)
    g = gate()
    g.limit("M", SELL, 100, 10)
    g.limit("T", BUY, 100, 4)
    assert [e[:5] for e in g.book.events] == [e[:5] for e in plain.events], \
        "risk checks must gate orders, never alter how they match"

    # rejects are recorded for audit
    g = gate(max_order_qty=1)
    try:
        g.limit("A", BUY, 100, 99)
    except RiskReject:
        pass
    assert g.rejects and g.rejects[0][1] == "MAX_ORDER_QTY", g.rejects

    print("ok — size, value, collar, position, open-order, self-trade checks pass")


if __name__ == "__main__":
    demo()
