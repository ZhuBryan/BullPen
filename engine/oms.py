"""
oms.py — order management: lifecycle, execution reports, positions and PnL.

The matching engine knows about orders. It does not know about *clients*, and
that gap is what an OMS fills. It owns:

  CLIENT ORDER IDS. A venue assigns its own OrderID; the client tracks its own
  ClOrdID. Cancel and replace are addressed by OrigClOrdID, not by the venue's
  id, because the client may not have received the ack yet.

  CANCEL/REPLACE. Amending an order is not an edit. It is a cancel of the old
  order and a new one in its place, with a fresh ClOrdID, and the replacement
  goes to the BACK of the queue at its price. Anyone who has watched a replace
  silently lose queue priority remembers this.

  EXECUTION REPORTS. Every state change emits one. That stream, not the order
  object, is the source of truth, because it is what a client actually sees.

  POSITION AND PnL. Average-cost accounting: realized PnL crystallises only
  when a position is reduced, unrealized is the rest marked to market. The
  subtle case is a position crossing through zero, where part of the fill
  closes the old side and the remainder opens the new one at the fill price.
  Treating that as one blended trade silently misstates both numbers.

Run `python oms.py` for the self-check.
"""

try:
    from engine import fix
    from engine.orderbook import BUY, SELL, Book
except ImportError:
    import fix
    from orderbook import BUY, SELL, Book

# Book sides are "B"/"S"; FIX uses "1"/"2". Convert once, at the boundary.
FIX_TO_BOOK = {fix.Side.BUY: BUY, fix.Side.SELL: SELL}
BOOK_TO_FIX = {BUY: fix.Side.BUY, SELL: fix.Side.SELL}


class Position:
    """Average-cost position accounting for a single symbol."""

    def __init__(self, symbol):
        self.symbol = symbol
        self.qty = 0            # signed: positive long, negative short
        self.avg_px = 0.0       # average cost of the OPEN position only
        self.realized = 0.0
        self.fees = 0.0

    def apply(self, side, qty, price, fee=0.0):
        """Apply a fill. Returns the realized PnL produced by THIS fill."""
        signed = qty if side == BUY else -qty
        self.fees += fee
        realized = 0.0

        if self.qty == 0 or (self.qty > 0) == (signed > 0):
            # Opening or adding: blend into the average cost. No PnL yet.
            total = self.qty + signed
            self.avg_px = (self.avg_px * abs(self.qty) + price * qty) / abs(total)
            self.qty = total
        else:
            # Reducing, closing, or flipping through zero.
            closing = min(abs(signed), abs(self.qty))
            # Long closed above cost is a gain; short closed below cost is too.
            direction = 1 if self.qty > 0 else -1
            realized = (price - self.avg_px) * closing * direction
            self.realized += realized
            remaining = abs(signed) - closing
            self.qty += signed
            if self.qty == 0:
                self.avg_px = 0.0
            elif remaining > 0:
                # Flipped sides: the remainder opens a new position at the
                # fill price. Blending it with the old average would be wrong.
                self.avg_px = price
        return realized

    def unrealized(self, mark):
        if self.qty == 0:
            return 0.0
        return (mark - self.avg_px) * self.qty

    def total(self, mark):
        return self.realized + self.unrealized(mark) - self.fees

    def snapshot(self, mark=None):
        d = {"symbol": self.symbol, "qty": self.qty,
             "avg_px": round(self.avg_px, 6),
             "realized": round(self.realized, 6),
             "fees": round(self.fees, 6)}
        if mark is not None:
            d["unrealized"] = round(self.unrealized(mark), 6)
            d["total"] = round(self.total(mark), 6)
        return d


class OMS:
    """Client-facing order management on top of a matching engine."""

    def __init__(self, book=None, symbol="TEST", fee_per_share=0.0, risk=None,
                 metrics=None):
        self.book = book or Book()
        self.symbol = symbol
        self.fee_per_share = fee_per_share
        self.risk = risk                      # optional RiskGate
        self.metrics = metrics                # optional Metrics, see metrics.py
        self.orders: dict[str, fix.OrderState] = {}   # ClOrdID -> state
        self.by_venue_id: dict[int, str] = {}         # book order id -> ClOrdID
        self.reports: list[dict] = []
        self.position = Position(symbol)
        self._exec_seq = 0
        self._seen_events = 0

    # ---- reporting -----------------------------------------------------
    def _exec_id(self):
        self._exec_seq += 1
        return f"EX{self._exec_seq}"

    def _report(self, order, exec_type, text=None, last_qty=None, last_px=None):
        rpt = {
            "exec_id": self._exec_id(),
            "cl_ord_id": order.cl_ord_id,
            "symbol": order.symbol,
            "side": order.side,
            "exec_type": exec_type,
            "ord_status": order.status,
            "leaves_qty": order.leaves_qty,
            "cum_qty": order.cum_qty,
            "avg_px": round(order.avg_px, 6),
        }
        if last_qty is not None:
            rpt["last_qty"], rpt["last_px"] = last_qty, last_px
        if text:
            rpt["text"] = text
        self.reports.append(rpt)
        return rpt

    def to_fix(self, rpt) -> str:
        """Render a report as a real ExecutionReport message."""
        return fix.execution_report(
            order_id=rpt.get("venue_id", "0"), exec_id=rpt["exec_id"],
            cl_ord_id=rpt["cl_ord_id"], symbol=rpt["symbol"], side=rpt["side"],
            exec_type=rpt["exec_type"], ord_status=rpt["ord_status"],
            leaves_qty=rpt["leaves_qty"], cum_qty=rpt["cum_qty"],
            avg_px=rpt["avg_px"], last_qty=rpt.get("last_qty"),
            last_px=rpt.get("last_px"), text=rpt.get("text"))

    # ---- order entry ---------------------------------------------------
    def new_order(self, cl_ord_id, side, qty, price=None, account="default"):
        """Accept a NewOrderSingle. Returns the reports it generated."""
        import time
        t0 = time.perf_counter_ns()
        start = len(self.reports)
        if cl_ord_id in self.orders:
            raise fix.FixError(f"duplicate ClOrdID {cl_ord_id}")

        order = fix.OrderState(cl_ord_id, self.symbol, side, qty, price)
        self.orders[cl_ord_id] = order

        if self.risk is not None:
            try:
                self._risk_precheck(account, side, price, qty)
            except Exception as e:
                order.transition(fix.OrdStatus.REJECTED)
                self._report(order, fix.ExecType.REJECTED, text=str(e))
                self._count("orders.rejected")
                return self.reports[start:]

        book_side = FIX_TO_BOOK[side] if side in FIX_TO_BOOK else side
        venue_id = (self.book.market(book_side, qty) if price is None
                    else self.book.limit(book_side, price, qty))
        self.by_venue_id[venue_id] = cl_ord_id
        order.transition(fix.OrdStatus.NEW)
        self._report(order, fix.ExecType.NEW)["venue_id"] = venue_id

        self._drain()
        self._count("orders.accepted")
        if self.metrics is not None:
            # Order-to-ack: submission through matching to the first report.
            self.metrics.observe("order.ack_latency", time.perf_counter_ns() - t0)
        # A market order that could not fill completely never rests.
        if price is None and not order.is_terminal and order.leaves_qty > 0:
            order.transition(fix.OrdStatus.CANCELED)
            self._report(order, fix.ExecType.CANCELED, text="IOC remainder")
        return self.reports[start:]

    def _risk_precheck(self, account, side, price, qty):
        book_side = FIX_TO_BOOK.get(side, side)
        self.risk._check(account, book_side, price, qty)

    def cancel(self, cl_ord_id, orig_cl_ord_id):
        """OrderCancelRequest. Addressed by the ORIGINAL ClOrdID."""
        start = len(self.reports)
        order = self.orders.get(orig_cl_ord_id)
        if order is None or order.is_terminal:
            # A cancel for an unknown or finished order gets a CancelReject,
            # not an ExecutionReport. They are different messages on the wire.
            self.reports.append({"exec_id": self._exec_id(), "cl_ord_id": cl_ord_id,
                                 "orig_cl_ord_id": orig_cl_ord_id,
                                 "exec_type": "CANCEL_REJECT",
                                 "text": "unknown or already terminal order"})
            return self.reports[start:]

        order.transition(fix.OrdStatus.PENDING_CANCEL)
        venue_id = next((v for v, c in self.by_venue_id.items() if c == orig_cl_ord_id), None)
        if venue_id is not None:
            self.book.cancel(venue_id)
        order.transition(fix.OrdStatus.CANCELED)
        self._report(order, fix.ExecType.CANCELED)
        self._count("orders.canceled")
        return self.reports[start:]

    def replace(self, cl_ord_id, orig_cl_ord_id, qty, price, account="default"):
        """OrderCancelReplaceRequest: cancel the old, submit a new one.

        The replacement goes to the back of the queue at its price. That is not
        an implementation shortcut, it is how venues behave, and a strategy that
        amends frequently pays for it in lost priority.
        """
        start = len(self.reports)
        order = self.orders.get(orig_cl_ord_id)
        if order is None or order.is_terminal:
            self.reports.append({"exec_id": self._exec_id(), "cl_ord_id": cl_ord_id,
                                 "orig_cl_ord_id": orig_cl_ord_id,
                                 "exec_type": "CANCEL_REJECT",
                                 "text": "cannot replace a terminal order"})
            return self.reports[start:]

        already = order.cum_qty
        if qty <= already:
            self.reports.append({"exec_id": self._exec_id(), "cl_ord_id": cl_ord_id,
                                 "orig_cl_ord_id": orig_cl_ord_id,
                                 "exec_type": "CANCEL_REJECT",
                                 "text": f"new qty {qty} below filled {already}"})
            return self.reports[start:]

        order.transition(fix.OrdStatus.PENDING_REPLACE)
        venue_id = next((v for v, c in self.by_venue_id.items() if c == orig_cl_ord_id), None)
        if venue_id is not None:
            self.book.cancel(venue_id)
        order.transition(fix.OrdStatus.REPLACED)
        self._report(order, fix.ExecType.REPLACED, text=f"replaced by {cl_ord_id}")

        # The new order carries over what the old one already did.
        self.new_order(cl_ord_id, order.side, qty - already, price, account)
        return self.reports[start:]

    def _count(self, name, by=1):
        if self.metrics is not None:
            self.metrics.incr(name, by)

    # ---- fills ---------------------------------------------------------
    def _drain(self):
        """Turn new book trades into fills, reports and position changes."""
        for ev in self.book.events[self._seen_events:]:
            if ev[0] != "trade":
                continue
            _, price, qty, aggressor, resting = ev[:5]
            for venue_id in (aggressor, resting):
                cl = self.by_venue_id.get(venue_id)
                if cl is None:
                    continue
                order = self.orders[cl]
                if order.is_terminal:
                    continue
                order.apply_fill(qty, price)
                fee = self.fee_per_share * qty
                book_side = FIX_TO_BOOK.get(order.side, order.side)
                self.position.apply(book_side, qty, price, fee)
                self._report(order, fix.ExecType.TRADE,
                             last_qty=qty, last_px=price)
                self._count("fills")
                self._count("shares.filled", qty)
        self._seen_events = len(self.book.events)

    # ---- views ---------------------------------------------------------
    def open_orders(self):
        return [o for o in self.orders.values() if not o.is_terminal]

    def blotter(self, mark=None):
        return {
            "orders": [{"cl_ord_id": o.cl_ord_id, "side": o.side, "qty": o.qty,
                        "price": o.price, "status": o.status,
                        "cum_qty": o.cum_qty, "leaves_qty": o.leaves_qty,
                        "avg_px": round(o.avg_px, 6)}
                       for o in self.orders.values()],
            "position": self.position.snapshot(mark),
            "reports": len(self.reports),
        }


def demo():
    # ---- position accounting, the part that is easy to get wrong ----
    p = Position("T")
    p.apply(BUY, 100, 10.0)
    assert p.qty == 100 and p.avg_px == 10.0 and p.realized == 0.0

    p.apply(BUY, 100, 12.0)                      # add: blended cost, no PnL
    assert p.qty == 200 and abs(p.avg_px - 11.0) < 1e-9 and p.realized == 0.0

    r = p.apply(SELL, 50, 15.0)                  # reduce: realize on 50 only
    assert abs(r - (15.0 - 11.0) * 50) < 1e-9, r
    assert p.qty == 150 and abs(p.avg_px - 11.0) < 1e-9, "cost basis must not move"

    # unrealized marks the rest
    assert abs(p.unrealized(12.0) - (12.0 - 11.0) * 150) < 1e-9

    # flip through zero: close 150 long, open 50 short at the fill price
    p2 = Position("T")
    p2.apply(BUY, 100, 10.0)
    r = p2.apply(SELL, 150, 12.0)
    assert abs(r - (12.0 - 10.0) * 100) < 1e-9, "only the closed 100 realizes"
    assert p2.qty == -50, p2.qty
    assert abs(p2.avg_px - 12.0) < 1e-9, "new short opens at the fill price"

    # shorts: profit when covered lower
    p3 = Position("T")
    p3.apply(SELL, 100, 20.0)
    assert p3.qty == -100 and abs(p3.avg_px - 20.0) < 1e-9
    r = p3.apply(BUY, 100, 18.0)
    assert abs(r - (18.0 - 20.0) * 100 * -1) < 1e-9, "covering lower is a gain"
    assert p3.qty == 0 and p3.avg_px == 0.0

    # fees reduce total but never touch realized
    p4 = Position("T")
    p4.apply(BUY, 100, 10.0, fee=1.0)
    p4.apply(SELL, 100, 11.0, fee=1.0)
    assert abs(p4.realized - 100.0) < 1e-9 and abs(p4.fees - 2.0) < 1e-9
    assert abs(p4.total(11.0) - 98.0) < 1e-9

    # ---- order lifecycle against a real book ----
    oms = OMS(symbol="AAPL")
    oms.new_order("MAKER", fix.Side.SELL, 100, price=10_000)
    rpts = oms.new_order("TAKER", fix.Side.BUY, 60, price=10_000)
    kinds = [r["exec_type"] for r in rpts]
    assert fix.ExecType.NEW in kinds and fix.ExecType.TRADE in kinds, kinds

    taker = oms.orders["TAKER"]
    assert taker.status == fix.OrdStatus.FILLED and taker.cum_qty == 60
    maker = oms.orders["MAKER"]
    assert maker.status == fix.OrdStatus.PARTIALLY_FILLED and maker.leaves_qty == 40

    # both sides of the fill hit the position, so it nets flat here
    assert oms.position.qty == 0, oms.position.snapshot()

    # every report renders as a valid FIX message
    for r in oms.reports:
        if r["exec_type"] != "CANCEL_REJECT":
            fix.decode(oms.to_fix(r))

    # ---- cancel ----
    out = oms.cancel("CXL1", "MAKER")
    assert out[-1]["exec_type"] == fix.ExecType.CANCELED
    assert oms.orders["MAKER"].status == fix.OrdStatus.CANCELED
    assert oms.book.best_ask() is None, "cancel must reach the book"

    # cancelling again is a CancelReject, not an ExecutionReport
    out = oms.cancel("CXL2", "MAKER")
    assert out[-1]["exec_type"] == "CANCEL_REJECT", out

    # ---- replace loses queue priority ----
    o2 = OMS(symbol="AAPL")
    first = o2.new_order("A1", fix.Side.BUY, 50, price=9_900)
    o2.new_order("B1", fix.Side.BUY, 50, price=9_900)     # behind A1
    o2.replace("A2", "A1", qty=50, price=9_900)           # A1 -> back of queue
    assert o2.orders["A1"].status == fix.OrdStatus.REPLACED
    # a seller crossing 50 should now hit B1, not the replaced order
    o2.new_order("S1", fix.Side.SELL, 50, price=9_900)
    assert o2.orders["B1"].status == fix.OrdStatus.FILLED, "replace kept priority"
    assert o2.orders["A2"].cum_qty == 0

    # replacing below the filled quantity is refused
    o3 = OMS(symbol="AAPL")
    o3.new_order("M", fix.Side.SELL, 100, price=100)
    o3.new_order("T", fix.Side.BUY, 60, price=100)        # M partially filled 60
    out = o3.replace("M2", "M", qty=50, price=100)
    assert out[-1]["exec_type"] == "CANCEL_REJECT", out

    # duplicate ClOrdID is refused
    try:
        o3.new_order("M", fix.Side.SELL, 1, price=100)
        raise AssertionError("duplicate ClOrdID accepted")
    except fix.FixError:
        pass

    # ---- market order remainder is cancelled, never rested ----
    o4 = OMS(symbol="AAPL")
    o4.new_order("LIQ", fix.Side.SELL, 10, price=100)
    rpts = o4.new_order("MKT", fix.Side.BUY, 25, price=None)
    assert o4.orders["MKT"].cum_qty == 10
    assert o4.orders["MKT"].status == fix.OrdStatus.CANCELED, "remainder must not rest"
    assert o4.book.best_bid() is None

    # ---- PnL through the OMS, marked to market ----
    o5 = OMS(symbol="AAPL", fee_per_share=0.01)
    o5.new_order("S", fix.Side.SELL, 100, price=50)
    o5.new_order("B", fix.Side.BUY, 100, price=50)
    snap = o5.position.snapshot(mark=55)
    assert snap["qty"] == 0 and abs(snap["fees"] - 2.0) < 1e-9, snap

    # ---- metrics are wired, not decorative ----
    try:
        from engine.metrics import Metrics
    except ImportError:
        from metrics import Metrics
    m = Metrics()
    om = OMS(symbol="AAPL", metrics=m)
    om.new_order("M", fix.Side.SELL, 100, price=100)
    om.new_order("T", fix.Side.BUY, 40, price=100)
    om.cancel("C", "M")
    snap = m.snapshot()
    assert snap["counters"]["orders.accepted"] == 2, snap
    assert snap["counters"]["orders.canceled"] == 1
    assert snap["counters"]["fills"] == 2, "both sides of a fill count"
    assert snap["counters"]["shares.filled"] == 80, snap
    assert snap["latency"]["order.ack_latency"]["count"] == 2
    assert "orders_accepted 2" in m.render_prometheus()

    print("ok — FIX lifecycle, cancel/replace priority, IOC remainder, "
          "average-cost PnL incl. flip through zero")


if __name__ == "__main__":
    demo()
