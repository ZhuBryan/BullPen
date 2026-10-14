"""
fix.py — FIX 4.4 order entry: encoding, decoding, and the order state machine.

FIX is how orders actually reach a venue. Equities, futures and options order
entry is overwhelmingly FIX, so an engine that only accepts a Python function
call is missing the layer every real system has. This module is the wire format
and the lifecycle rules, with no sockets: a FIX session's framing and its
business logic are separate problems, and only the second one is interesting
here.

Three things people get wrong and that the tests below pin down:

  BODYLENGTH is not the length of the message. It counts the bytes after the
  `9=...<SOH>` field up to and including the <SOH> before the checksum field.
  Off by one here and every counterparty rejects you.

  CHECKSUM is the sum of every raw byte of the message up to and including the
  <SOH> before `10=`, modulo 256, rendered as exactly three digits. Leading
  zeros matter: "7" is wrong, "007" is right.

  FIELD ORDER is part of the protocol, not a detail. 8, 9 must lead and 10 must
  trail. A dict that happens to serialize in the right order on your machine is
  not a guarantee.

The order state machine is the other half. An order is not a boolean. It moves
New -> PartiallyFilled -> Filled, or out to Canceled / Rejected / Replaced, and
a venue that permits an illegal transition (filling a cancelled order, say)
will eventually trade something a client already stood down.

Run `python fix.py` for the self-check.
"""

SOH = "\x01"
BEGIN_STRING = "FIX.4.4"

# --- the tags this module speaks, by name so call sites stay readable --------
TAGS = {
    "BeginString": 8, "BodyLength": 9, "CheckSum": 10, "MsgType": 35,
    "SenderCompID": 49, "TargetCompID": 56, "MsgSeqNum": 34, "SendingTime": 52,
    "ClOrdID": 11, "OrigClOrdID": 41, "OrderID": 37, "ExecID": 17,
    "Symbol": 55, "Side": 54, "OrderQty": 38, "OrdType": 40, "Price": 44,
    "TimeInForce": 59, "ExecType": 150, "OrdStatus": 39, "LastQty": 32,
    "LastPx": 31, "LeavesQty": 151, "CumQty": 14, "AvgPx": 6,
    "Text": 58, "OrdRejReason": 103, "CxlRejReason": 102,
    "TransactTime": 60, "Account": 1,
}
NAME_BY_TAG = {v: k for k, v in TAGS.items()}

# --- enumerations, using the real FIX values --------------------------------
class MsgType:
    LOGON = "A"
    HEARTBEAT = "0"
    REJECT = "3"
    EXECUTION_REPORT = "8"
    ORDER_CANCEL_REJECT = "9"
    NEW_ORDER_SINGLE = "D"
    ORDER_CANCEL_REQUEST = "F"
    ORDER_CANCEL_REPLACE_REQUEST = "G"


class Side:
    BUY = "1"
    SELL = "2"


class OrdType:
    MARKET = "1"
    LIMIT = "2"


class TimeInForce:
    DAY = "0"
    IOC = "3"
    FOK = "4"


class OrdStatus:
    NEW = "0"
    PARTIALLY_FILLED = "1"
    FILLED = "2"
    CANCELED = "4"
    PENDING_CANCEL = "6"
    REJECTED = "8"
    PENDING_NEW = "A"
    PENDING_REPLACE = "E"
    REPLACED = "5"


class ExecType:
    NEW = "0"
    TRADE = "F"
    CANCELED = "4"
    REPLACED = "5"
    REJECTED = "8"
    PENDING_CANCEL = "6"


class FixError(ValueError):
    """Malformed message. The text is safe to send back as a Reject."""


# ---------------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------------
def checksum(payload: str) -> str:
    """Sum of every byte modulo 256, always three digits."""
    return f"{sum(payload.encode('ascii')) % 256:03d}"


def encode(msg_type: str, fields, sender="BULLPEN", target="VENUE", seq=1) -> str:
    """Build a complete FIX message from (tag, value) pairs.

    `fields` is an ordered sequence so the caller controls field order, which
    is part of the protocol rather than an implementation detail.
    """
    header = [(TAGS["MsgType"], msg_type),
              (TAGS["SenderCompID"], sender),
              (TAGS["TargetCompID"], target),
              (TAGS["MsgSeqNum"], seq)]
    body = "".join(f"{t}={v}{SOH}" for t, v in list(header) + list(fields))
    # BodyLength covers everything after 9=...<SOH>, up to the <SOH> before 10=.
    head = f"{TAGS['BeginString']}={BEGIN_STRING}{SOH}{TAGS['BodyLength']}={len(body)}{SOH}"
    return head + body + f"{TAGS['CheckSum']}={checksum(head + body)}{SOH}"


def decode(raw: str, verify=True) -> dict:
    """Parse a FIX message into {'pairs': [(tag, value)], 'fields': {tag: value}}.

    Raises FixError on a bad checksum or body length, because silently
    accepting a corrupt message is how a venue ends up acting on garbage.
    """
    if not raw.endswith(SOH):
        raise FixError("message does not end with SOH")
    parts = [p for p in raw.split(SOH) if p]
    pairs = []
    for p in parts:
        if "=" not in p:
            raise FixError(f"field without '=': {p!r}")
        tag, _, value = p.partition("=")
        if not tag.isdigit():
            raise FixError(f"non-numeric tag: {tag!r}")
        pairs.append((int(tag), value))

    if verify:
        if not pairs or pairs[0][0] != TAGS["BeginString"]:
            raise FixError("missing BeginString(8) as first field")
        if len(pairs) < 2 or pairs[1][0] != TAGS["BodyLength"]:
            raise FixError("missing BodyLength(9) as second field")
        if pairs[-1][0] != TAGS["CheckSum"]:
            raise FixError("missing CheckSum(10) as last field")

        idx = raw.index(f"{SOH}{TAGS['CheckSum']}=") + 1
        if checksum(raw[:idx]) != pairs[-1][1]:
            raise FixError(f"checksum mismatch: expected {checksum(raw[:idx])}, "
                           f"got {pairs[-1][1]}")
        body_start = raw.index(SOH, raw.index(f"{TAGS['BodyLength']}=")) + 1
        if int(pairs[1][1]) != idx - body_start:
            raise FixError(f"body length mismatch: declared {pairs[1][1]}, "
                           f"actual {idx - body_start}")

    return {"pairs": pairs, "fields": {t: v for t, v in pairs}}


def get(msg: dict, name: str, default=None):
    """Field lookup by human name rather than magic number."""
    return msg["fields"].get(TAGS[name], default)


# --- message builders -------------------------------------------------------
def new_order_single(cl_ord_id, symbol, side, qty, ord_type=OrdType.LIMIT,
                     price=None, tif=TimeInForce.DAY, account=None, **kw):
    f = [(TAGS["ClOrdID"], cl_ord_id), (TAGS["Symbol"], symbol),
         (TAGS["Side"], side), (TAGS["OrderQty"], qty),
         (TAGS["OrdType"], ord_type), (TAGS["TimeInForce"], tif)]
    if account:
        f.insert(0, (TAGS["Account"], account))
    if ord_type == OrdType.LIMIT:
        if price is None:
            raise FixError("a limit order requires a price")
        f.append((TAGS["Price"], price))
    return encode(MsgType.NEW_ORDER_SINGLE, f, **kw)


def execution_report(order_id, exec_id, cl_ord_id, symbol, side, exec_type,
                     ord_status, leaves_qty, cum_qty, avg_px,
                     last_qty=None, last_px=None, text=None, **kw):
    f = [(TAGS["OrderID"], order_id), (TAGS["ExecID"], exec_id),
         (TAGS["ClOrdID"], cl_ord_id), (TAGS["Symbol"], symbol),
         (TAGS["Side"], side), (TAGS["ExecType"], exec_type),
         (TAGS["OrdStatus"], ord_status), (TAGS["LeavesQty"], leaves_qty),
         (TAGS["CumQty"], cum_qty), (TAGS["AvgPx"], avg_px)]
    if last_qty is not None:
        f += [(TAGS["LastQty"], last_qty), (TAGS["LastPx"], last_px)]
    if text:
        f.append((TAGS["Text"], text))
    return encode(MsgType.EXECUTION_REPORT, f, **kw)


def cancel_request(cl_ord_id, orig_cl_ord_id, symbol, side, qty, **kw):
    return encode(MsgType.ORDER_CANCEL_REQUEST,
                  [(TAGS["ClOrdID"], cl_ord_id),
                   (TAGS["OrigClOrdID"], orig_cl_ord_id),
                   (TAGS["Symbol"], symbol), (TAGS["Side"], side),
                   (TAGS["OrderQty"], qty)], **kw)


def cancel_replace_request(cl_ord_id, orig_cl_ord_id, symbol, side, qty,
                           price, ord_type=OrdType.LIMIT, **kw):
    return encode(MsgType.ORDER_CANCEL_REPLACE_REQUEST,
                  [(TAGS["ClOrdID"], cl_ord_id),
                   (TAGS["OrigClOrdID"], orig_cl_ord_id),
                   (TAGS["Symbol"], symbol), (TAGS["Side"], side),
                   (TAGS["OrderQty"], qty), (TAGS["OrdType"], ord_type),
                   (TAGS["Price"], price)], **kw)


def reject(ref_seq, reason, text=None, **kw):
    f = [(45, ref_seq), (TAGS["Text"], text or reason)]
    return encode(MsgType.REJECT, f, **kw)


# ---------------------------------------------------------------------------
# Order state machine
# ---------------------------------------------------------------------------
# An order is not a boolean. Allowing an illegal transition means a venue can
# fill something the client already cancelled, so the legal moves are explicit.
TERMINAL = {OrdStatus.FILLED, OrdStatus.CANCELED, OrdStatus.REJECTED,
            OrdStatus.REPLACED}

TRANSITIONS = {
    OrdStatus.PENDING_NEW: {OrdStatus.NEW, OrdStatus.REJECTED},
    OrdStatus.NEW: {OrdStatus.PARTIALLY_FILLED, OrdStatus.FILLED,
                    OrdStatus.CANCELED, OrdStatus.PENDING_CANCEL,
                    OrdStatus.PENDING_REPLACE, OrdStatus.REJECTED},
    OrdStatus.PARTIALLY_FILLED: {OrdStatus.PARTIALLY_FILLED, OrdStatus.FILLED,
                                 OrdStatus.CANCELED, OrdStatus.PENDING_CANCEL,
                                 OrdStatus.PENDING_REPLACE},
    OrdStatus.PENDING_CANCEL: {OrdStatus.CANCELED, OrdStatus.PARTIALLY_FILLED,
                               OrdStatus.FILLED, OrdStatus.NEW},
    OrdStatus.PENDING_REPLACE: {OrdStatus.REPLACED, OrdStatus.NEW,
                                OrdStatus.PARTIALLY_FILLED, OrdStatus.FILLED},
    # terminal states go nowhere
    OrdStatus.FILLED: set(), OrdStatus.CANCELED: set(),
    OrdStatus.REJECTED: set(), OrdStatus.REPLACED: set(),
}


def can_transition(frm: str, to: str) -> bool:
    return to in TRANSITIONS.get(frm, set())


class OrderState:
    """Tracks one order's lifecycle and fill arithmetic."""

    def __init__(self, cl_ord_id, symbol, side, qty, price=None):
        self.cl_ord_id = cl_ord_id
        self.symbol, self.side = symbol, side
        self.qty = int(qty)
        self.price = price
        self.status = OrdStatus.PENDING_NEW
        self.cum_qty = 0
        self.avg_px = 0.0
        self.history = [OrdStatus.PENDING_NEW]

    @property
    def leaves_qty(self):
        return 0 if self.status in TERMINAL else self.qty - self.cum_qty

    @property
    def is_terminal(self):
        return self.status in TERMINAL

    def transition(self, to: str):
        if not can_transition(self.status, to):
            raise FixError(f"illegal transition {self.status} -> {to} "
                           f"for {self.cl_ord_id}")
        self.status = to
        self.history.append(to)
        return self

    def apply_fill(self, qty, price):
        """Apply an execution, updating cumulative quantity and average price."""
        qty = int(qty)
        if qty <= 0:
            raise FixError("fill quantity must be positive")
        if self.is_terminal:
            raise FixError(f"cannot fill a {self.status} order")
        if self.cum_qty + qty > self.qty:
            raise FixError(f"overfill: {self.cum_qty}+{qty} exceeds {self.qty}")
        # Average price is quantity-weighted across every execution, which is
        # what AvgPx(6) means on the wire.
        self.avg_px = (self.avg_px * self.cum_qty + price * qty) / (self.cum_qty + qty)
        self.cum_qty += qty
        self.transition(OrdStatus.FILLED if self.cum_qty == self.qty
                        else OrdStatus.PARTIALLY_FILLED)
        return self


def demo():
    # ---- wire format ----
    msg = new_order_single("ORD1", "AAPL", Side.BUY, 100, price=185.25)
    assert msg.startswith(f"8={BEGIN_STRING}{SOH}9=")
    assert msg.endswith(SOH)
    d = decode(msg)
    assert get(d, "ClOrdID") == "ORD1" and get(d, "Symbol") == "AAPL"
    assert get(d, "Side") == Side.BUY and get(d, "OrderQty") == "100"
    assert get(d, "Price") == "185.25"
    assert d["pairs"][0][0] == 8 and d["pairs"][1][0] == 9, "8 and 9 must lead"
    assert d["pairs"][-1][0] == 10, "checksum must trail"

    # checksum is three digits, zero padded
    assert len(get(d, "CheckSum")) == 3
    assert checksum("A") == "065", checksum("A")        # ord('A') == 65
    assert checksum(SOH * 3) == "003", "must zero-pad"

    # body length counts the bytes between 9=...SOH and the SOH before 10=
    body_declared = int(get(d, "BodyLength"))
    start = msg.index(SOH, msg.index("9=")) + 1
    end = msg.index(f"{SOH}10=") + 1
    assert body_declared == end - start, "BodyLength must match the payload"

    # ---- corruption is rejected, not silently accepted ----
    for broken, why in [
        (msg.replace("ORD1", "ORD2"), "checksum should fail after tampering"),
        (msg[:-1], "missing trailing SOH"),
        (msg.replace(f"9={body_declared}", "9=999"), "bad body length"),
    ]:
        try:
            decode(broken)
            raise AssertionError(why)
        except FixError:
            pass

    # a market order must not carry a price, a limit order must
    try:
        new_order_single("X", "AAPL", Side.BUY, 10, ord_type=OrdType.LIMIT)
        raise AssertionError("limit order without a price should raise")
    except FixError:
        pass
    mkt = decode(new_order_single("M1", "AAPL", Side.BUY, 10, ord_type=OrdType.MARKET))
    assert get(mkt, "Price") is None and get(mkt, "OrdType") == OrdType.MARKET

    # round trip of every builder
    for raw in [
        execution_report("V1", "E1", "ORD1", "AAPL", Side.BUY, ExecType.TRADE,
                         OrdStatus.PARTIALLY_FILLED, 60, 40, 185.2,
                         last_qty=40, last_px=185.2),
        cancel_request("C1", "ORD1", "AAPL", Side.BUY, 100),
        cancel_replace_request("R1", "ORD1", "AAPL", Side.BUY, 150, 185.5),
        reject(7, "bad field"),
    ]:
        decode(raw)   # raises if malformed

    er = decode(execution_report("V1", "E2", "ORD1", "AAPL", Side.BUY,
                                 ExecType.TRADE, OrdStatus.FILLED, 0, 100, 185.3,
                                 last_qty=60, last_px=185.4))
    assert get(er, "ExecType") == ExecType.TRADE
    assert get(er, "LeavesQty") == "0" and get(er, "CumQty") == "100"

    # ---- state machine ----
    o = OrderState("ORD1", "AAPL", Side.BUY, 100, 185.25)
    assert o.status == OrdStatus.PENDING_NEW and o.leaves_qty == 100
    o.transition(OrdStatus.NEW)
    o.apply_fill(40, 185.00)
    assert o.status == OrdStatus.PARTIALLY_FILLED and o.leaves_qty == 60
    o.apply_fill(60, 185.50)
    assert o.status == OrdStatus.FILLED and o.leaves_qty == 0
    # 40@185.00 + 60@185.50 -> 185.30
    assert abs(o.avg_px - 185.30) < 1e-9, o.avg_px

    # a filled order is terminal: no more fills, no cancel
    for bad in (lambda: o.apply_fill(1, 185.0),
                lambda: o.transition(OrdStatus.CANCELED)):
        try:
            bad(); raise AssertionError("terminal order accepted a transition")
        except FixError:
            pass

    # overfill is refused
    o2 = OrderState("ORD2", "AAPL", Side.SELL, 50)
    o2.transition(OrdStatus.NEW)
    try:
        o2.apply_fill(51, 1.0); raise AssertionError("overfill accepted")
    except FixError:
        pass

    # cancelling a working order, and the illegal moves around it
    o3 = OrderState("ORD3", "AAPL", Side.BUY, 10, 1.0)
    o3.transition(OrdStatus.NEW).transition(OrdStatus.PENDING_CANCEL)
    o3.transition(OrdStatus.CANCELED)
    assert o3.is_terminal and o3.leaves_qty == 0
    try:
        o3.apply_fill(1, 1.0)
        raise AssertionError("filled a cancelled order")
    except FixError:
        pass

    # rejection straight from PendingNew is legal; a fill from there is not
    o4 = OrderState("ORD4", "AAPL", Side.BUY, 10, 1.0)
    o4.transition(OrdStatus.REJECTED)
    assert o4.is_terminal
    assert not can_transition(OrdStatus.NEW, OrdStatus.REPLACED), \
        "replace must pass through PendingReplace"
    assert can_transition(OrdStatus.PENDING_REPLACE, OrdStatus.REPLACED)

    print("ok — FIX encode/decode, checksum, body length, builders, state machine")


if __name__ == "__main__":
    demo()
