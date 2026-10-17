"""
fix_session.py — the FIX 4.4 session layer over TCP (asyncio, stdlib only).

fix.py is the wire format; oms.py is the business logic. This is the part in
between that real venues insist on: a session that guarantees every message is
delivered exactly once, in order, between two parties who both know who the
other one is. TCP alone does not give you that. A connection can drop with
bytes in flight, a process can restart, and the counterparty needs a way to
say "I missed 41 to 44, send them again". So FIX layers its own rules on top:

  SEQUENCE NUMBERS. Every message in each direction carries MsgSeqNum(34),
  incrementing by one. A gap means you lost something; a number that goes
  backwards means the other side is confused about state, which is serious
  enough that the spec says log out rather than guess.

  RESEND. A gap is repaired by ResendRequest(2). The sender replays business
  messages flagged PossDupFlag(43)=Y, so the receiver knows "you may already
  have acted on this", and replaces admin messages with a SequenceReset-GapFill.
  Replaying a stale heartbeat is meaningless; replaying a stale order without
  a flag is dangerous, which is exactly why PossDupFlag exists.

  HEARTBEATS. Silence is ambiguous: a quiet market or a dead peer? Each side
  sends Heartbeat(0) when idle, and probes with TestRequest(1) when the other
  side goes quiet. No answer means disconnect, because trading against a
  counterparty you cannot hear from is how orders get stranded.

Deliberately out of scope: encryption (EncryptMethod=0 only), persistence
across restarts (the message store is in memory), session schedules, and
full field/repeating-group validation. A production engine would add a
persistent message store (so seq numbers and resends survive a restart),
TLS, scheduled session start/end with daily seq reset, and throttling.

Run `python fix_session.py` for the self-check (about 3 seconds).
"""

import asyncio
import time
from datetime import datetime, timezone

try:
    from engine import fix
    from engine.oms import OMS
except ImportError:
    import fix
    from oms import OMS

BEGIN = f"8={fix.BEGIN_STRING}{fix.SOH}".encode()
MAX_BODY = 1 << 16          # refuse absurd BodyLength values instead of allocating them
HEADER_TAGS = {8, 9, 10, 35, 49, 56, 34, 52, 43, 122}

# Session-level (admin) message types.
LOGON, HEARTBEAT, TEST_REQUEST, RESEND_REQUEST = "A", "0", "1", "2"
REJECT, SEQUENCE_RESET, LOGOUT = "3", "4", "5"
# Admin messages that are never replayed on resend, only gap-filled. Reject is
# admin too but the spec says it IS resent: it tells the peer something useful.
NO_RESEND = {LOGON, HEARTBEAT, TEST_REQUEST, RESEND_REQUEST, SEQUENCE_RESET, LOGOUT}


def utc_now():
    """SendingTime(52) format: UTC, millisecond precision."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S.%f")[:-3]


def body_of(raw):
    """Strip header and trailer from a message built by a fix.py builder, so the
    session can stamp its own seq/time onto it. Reuses the builders unchanged."""
    return [(t, v) for t, v in fix.decode(raw)["pairs"] if t not in HEADER_TAGS]


def show(raw):
    return raw.replace(fix.SOH, "|")


class Garbled(Exception):
    pass


async def read_frame(reader):
    """Read exactly one FIX message off a byte stream.

    TCP has no message boundaries, so the length lives in the message: read
    8=FIX.4.4|9=N|, then exactly N bytes, then the fixed 7-byte 10=xxx| trailer.
    Anything before a BeginString is noise; readuntil skips it, which is also
    how the reader resynchronises after a garbled frame.
    """
    skipped = await reader.readuntil(BEGIN)
    length = await reader.readuntil(fix.SOH.encode())
    if not (length.startswith(b"9=") and length[2:-1].isdigit()) or int(length[2:-1]) > MAX_BODY:
        raise Garbled(f"bad BodyLength field {length!r}")
    body = await reader.readexactly(int(length[2:-1]))
    trailer = await reader.readexactly(7)
    return (BEGIN + length + body + trailer).decode("latin-1"), len(skipped) - len(BEGIN)


class Session:
    """One side of a FIX session. Acceptor and Initiator differ only in who
    sends Logon first and what they do with application messages."""

    def __init__(self, sender, target, reader, writer, hb=30, on_app=None, verbose=False):
        self.sender, self.target = sender, target
        self.reader, self.writer = reader, writer
        self.hb = hb
        self.on_app = on_app or (lambda sess, msg: None)
        self.verbose = verbose
        self.out_seq = 1                 # next MsgSeqNum we will send
        self.in_seq = 1                  # next MsgSeqNum we expect
        self.store = {}                  # seq -> (msg_type, fields, sending_time), for resends
        self.queue = {}                  # out-of-order inbound seq -> (msg, already_handled)
        self.resend_pending = False
        self.logged_on = asyncio.Event()
        self.closed = asyncio.Event()
        self.close_reason = None
        self.logout_sent = False
        self.last_sent = self.last_recv = time.monotonic()
        self.test_req_id, self.test_req_at, self._test_n = None, None, 0
        self.received = []               # every valid inbound message, for tests/audit
        self.log = []
        self.garbled = 0
        # Test hooks. Real code never sets these.
        self.drop_next_app = 0           # silently drop N outbound app messages
        self.send_heartbeats = True      # False: stop idle heartbeats (still answer TestRequest)
        self.silent = False              # True: write nothing at all (a hung peer)
        self._tasks = []

    def _log(self, text):
        self.log.append(text)
        if self.verbose:
            print(f"[{self.sender}] {text}")

    def start(self):
        self._tasks = [asyncio.create_task(self._read_loop()),
                       asyncio.create_task(self._timer())]

    # ---- outbound -------------------------------------------------------
    def _write(self, raw):
        if self.silent or self.closed.is_set():
            return
        self.writer.write(raw.encode("latin-1"))
        self.last_sent = time.monotonic()

    def send(self, msg_type, fields=()):
        """Stamp seq + SendingTime, store for resend, write. Returns the seq."""
        if self.silent or self.closed.is_set():
            return None
        seq, now, fields = self.out_seq, utc_now(), list(fields)
        self.out_seq += 1
        self.store[seq] = (msg_type, fields, now)
        if self.drop_next_app and msg_type not in NO_RESEND:
            self.drop_next_app -= 1
            self._log(f"TEST HOOK: dropped outbound {msg_type} seq={seq}")
            return seq
        self._write(fix.encode(msg_type, [(52, now)] + fields, self.sender, self.target, seq))
        return seq

    def send_reject(self, ref_seq, text, reason=99):
        # SessionRejectReason(373); 99 = Other. Session Reject means "your
        # message was malformed", not "your order was refused".
        self.send(REJECT, body_of(fix.reject(ref_seq, text)) + [(373, reason)])

    def _gap_fill(self, seq, new_seq):
        self._write(fix.encode(SEQUENCE_RESET, [(52, utc_now()), (43, "Y"), (123, "Y"), (36, new_seq)],
                               self.sender, self.target, seq))

    def _resend(self, begin, end):
        last = self.out_seq - 1
        end = last if end == 0 or end > last else end     # EndSeqNo 0 means "everything"
        self._log(f"resending {begin}..{end}")
        gap = None
        for seq in range(begin, end + 1):
            msg_type, fields, sent_at = self.store.get(seq, (HEARTBEAT, [], None))
            if msg_type in NO_RESEND:
                gap = seq if gap is None else gap        # collapse runs of admin msgs
                continue
            if gap is not None:
                self._gap_fill(gap, seq)
                gap = None
            # Same seq, PossDupFlag=Y, and the ORIGINAL sending time, so the
            # receiver can tell a replay from a fresh instruction.
            self._write(fix.encode(msg_type, [(52, utc_now()), (43, "Y"), (122, sent_at)] + fields,
                                   self.sender, self.target, seq))
        if gap is not None:
            self._gap_fill(gap, end + 1)

    async def logout(self, text=None, timeout=2.0):
        """Polite logout: send 5, wait for the peer's 5, then disconnect."""
        self.logout_sent = True
        self.send(LOGOUT, [(58, text)] if text else [])
        try:
            await asyncio.wait_for(self.closed.wait(), timeout)
        except asyncio.TimeoutError:
            self.close("logout not acknowledged")

    def _fatal(self, text):
        """Serious session error: tell the peer why, then hang up."""
        self._log(f"fatal: {text}")
        self.logout_sent = True
        self.send(LOGOUT, [(58, text)])
        self.close(text)

    def close(self, reason):
        if self.closed.is_set():
            return
        self.close_reason = reason
        self._log(f"closed: {reason}")
        self.closed.set()
        self.writer.close()          # transport flushes buffered bytes first
        for t in self._tasks:
            if t is not asyncio.current_task():
                t.cancel()

    # ---- inbound --------------------------------------------------------
    async def _read_loop(self):
        try:
            while not self.closed.is_set():
                try:
                    raw, skipped = await read_frame(self.reader)
                    if skipped:
                        self._log(f"skipped {skipped} bytes of noise before BeginString")
                    msg = fix.decode(raw)
                except (Garbled, fix.FixError, ValueError) as e:
                    # Spec: a garbled message is ignored, not rejected, and does
                    # NOT consume a sequence number. If it was real, the next
                    # good message exposes the gap and resend recovers it.
                    self.garbled += 1
                    self._log(f"discarded garbled message: {e}")
                    continue
                self._on_message(msg)
        except (asyncio.IncompleteReadError, ConnectionError, asyncio.LimitOverrunError):
            pass
        finally:
            self.close(self.close_reason or "connection closed by peer")

    def _on_message(self, msg):
        f = msg["fields"]
        msg_type = f.get(35)
        self.last_recv = time.monotonic()
        self.test_req_at = None          # any traffic proves the peer is alive
        if not self.logged_on.is_set() and msg_type != LOGON:
            return self.close("first message was not Logon")
        if msg_type == LOGON and self.target is None:
            self._learn_peer(msg)
            if self.closed.is_set():
                return
        # CompIDs are the session's identity. A message addressed to someone
        # else, or from someone else, on this socket means misrouting.
        if f.get(49) != self.target or f.get(56) != self.sender:
            self.send_reject(f.get(34, 0), "CompID problem", reason=9)
            return self._fatal("CompID problem")
        try:
            seq = int(f[34])
        except (KeyError, ValueError):
            return self._fatal("MsgSeqNum missing or invalid")
        self.received.append(msg)
        self._log(f"recv {show(msg_type)} seq={seq} (expected {self.in_seq})")

        if msg_type == SEQUENCE_RESET and f.get(123) != "Y":
            # Reset mode: an administrative override, seq number of this
            # message itself is ignored by definition.
            self.in_seq = int(f[36])
            return
        if seq < self.in_seq:
            if f.get(43) == "Y":
                return               # a replay of something already processed
            return self._fatal(f"MsgSeqNum too low, expected {self.in_seq} but received {seq}")
        if seq > self.in_seq:
            # Logon and ResendRequest are acted on immediately even when out of
            # order: if both sides waited for their own gap to fill before
            # servicing the peer's ResendRequest, they would deadlock.
            handled = msg_type in (LOGON, RESEND_REQUEST)
            if handled:
                self._dispatch(msg)
            self.queue[seq] = (msg, handled)
            if not self.resend_pending:
                self.resend_pending = True
                self._log(f"gap: expected {self.in_seq}, got {seq}; requesting resend")
                self.send(RESEND_REQUEST, [(7, self.in_seq), (16, 0)])
            return
        self._apply(msg)
        while self.in_seq in self.queue:
            self._apply(*self.queue.pop(self.in_seq))
        for s in [s for s in self.queue if s < self.in_seq]:
            del self.queue[s]        # superseded by a GapFill jump
        if not self.queue:
            self.resend_pending = False

    def _apply(self, msg, handled=False):
        """Process an in-sequence message and advance the expected number."""
        f = msg["fields"]
        if f.get(35) == SEQUENCE_RESET:          # GapFill: skip straight to NewSeqNo
            new = int(f[36])
            if new <= self.in_seq:
                return self.send_reject(f[34], "NewSeqNo must increase", reason=5)
            self.in_seq = new
            return
        self.in_seq += 1
        if not handled:
            self._dispatch(msg)

    def _learn_peer(self, msg):
        raise NotImplementedError        # only the acceptor learns its peer

    def _dispatch(self, msg):
        f = msg["fields"]
        t = f.get(35)
        if t == LOGON:
            self._on_logon(msg)
        elif t == HEARTBEAT:
            if f.get(112) and f.get(112) == self.test_req_id:
                self._log(f"TestRequest {self.test_req_id} answered")
                self.test_req_id = None
        elif t == TEST_REQUEST:
            self.send(HEARTBEAT, [(112, f.get(112, ""))])   # echo the id: proof it was THIS probe
        elif t == RESEND_REQUEST:
            self._resend(int(f[7]), int(f[16]))
        elif t == REJECT:
            self._log(f"peer rejected our seq {f.get(45)}: {f.get(58)}")
        elif t == LOGOUT:
            if not self.logout_sent:
                self.logout_sent = True
                self.send(LOGOUT)
            self.close(f"logout: {f.get(58, 'peer requested')}")
        elif t == SEQUENCE_RESET:
            pass                                # handled in _apply
        else:
            self.on_app(self, msg)

    def _on_logon(self, msg):
        self.logged_on.set()

    # ---- timers ---------------------------------------------------------
    async def _timer(self):
        while not self.closed.is_set():
            # Fixed 100ms tick rather than hb/10: the acceptor only learns
            # HeartBtInt from the Logon, after this loop has started.
            await asyncio.sleep(0.1)
            if not self.logged_on.is_set():
                continue
            now = time.monotonic()
            if self.send_heartbeats and now - self.last_sent >= self.hb:
                self.send(HEARTBEAT)
            if self.test_req_at is None and now - self.last_recv >= self.hb * 1.2:
                # Grace of 20% over HeartBtInt allows for network jitter
                # before we start doubting the peer.
                self._test_n += 1
                self.test_req_id = f"TEST{self._test_n}"
                self.test_req_at = now
                self._log(f"silent for {now - self.last_recv:.2f}s, sending TestRequest")
                self.send(TEST_REQUEST, [(112, self.test_req_id)])
            elif self.test_req_at is not None and now - self.test_req_at >= self.hb:
                self.close("no response to TestRequest")


class AcceptorSession(Session):
    def __init__(self, venue, reader, writer):
        super().__init__(venue.comp_id, None, reader, writer, on_app=venue.on_app,
                         verbose=venue.verbose)
        self.venue = venue

    def _learn_peer(self, msg):
        f = msg["fields"]
        if f.get(56) != self.sender or (self.venue.clients and f.get(49) not in self.venue.clients):
            # Spec: an unauthenticated Logon gets no session, just a disconnect.
            return self.close(f"Logon refused for {f.get(49)}->{f.get(56)}")
        self.target = f.get(49)

    def _on_logon(self, msg):
        f = msg["fields"]
        if f.get(98, "0") != "0":
            return self._fatal("EncryptMethod not supported, use 98=0")
        self.hb = max(1, int(f.get(108, 30)))      # the initiator chooses the interval
        reset = f.get(141) == "Y"
        if reset:
            self.out_seq, self.store = 1, {}
        self.venue.sessions[self.target] = self
        self.send(LOGON, [(98, 0), (108, self.hb)] + ([(141, "Y")] if reset else []))
        self.logged_on.set()


class Acceptor:
    """The venue side: a TCP server whose sessions feed one OMS + matching engine."""

    def __init__(self, oms=None, comp_id="VENUE", clients=None, verbose=False):
        self.oms = oms or OMS(symbol="TEST")
        self.comp_id, self.clients, self.verbose = comp_id, clients, verbose
        self.sessions = {}            # client CompID -> AcceptorSession
        self.owner = {}               # ClOrdID -> client CompID, for routing reports
        self.venue_id = {}            # ClOrdID -> book order id, the OrderID(37) we report
        self.server = None

    async def start(self, host="127.0.0.1", port=0):
        self.server = await asyncio.start_server(self._accept, host, port)
        return self.server.sockets[0].getsockname()[1]

    async def _accept(self, reader, writer):
        s = AcceptorSession(self, reader, writer)
        s.start()
        await s.closed.wait()

    async def stop(self):
        for s in list(self.sessions.values()):
            s.close("acceptor stopping")
        self.server.close()
        await self.server.wait_closed()

    def on_app(self, sess, msg):
        f = msg["fields"]
        t, seq = f[35], f[34]
        try:
            if f.get(55) not in (None, self.oms.symbol):
                raise ValueError(f"unknown symbol {f.get(55)}")
            if t == fix.MsgType.NEW_ORDER_SINGLE:
                # Prices inside the engine are integer ticks, and so on this wire.
                price = None if f.get(40) == fix.OrdType.MARKET else int(f[44])
                self.owner[f[11]] = sess.target
                rpts = self.oms.new_order(f[11], f[54], int(f[38]), price, account=sess.target)
            elif t == fix.MsgType.ORDER_CANCEL_REQUEST:
                rpts = self.oms.cancel(f[11], f[41])
            elif t == fix.MsgType.ORDER_CANCEL_REPLACE_REQUEST:
                self.owner[f[11]] = sess.target
                rpts = self.oms.replace(f[11], f[41], int(f[38]), int(f[44]), account=sess.target)
            else:
                return sess.send_reject(seq, f"unsupported MsgType {t}", reason=11)
        except (KeyError, ValueError, fix.FixError) as e:
            # A resent order (PossDupFlag=Y) the OMS already has lands here as
            # "duplicate ClOrdID": that is the business-level dedup that makes
            # PossDup replays safe.
            return sess.send_reject(seq, f"{type(e).__name__}: {e}", reason=5)
        for r in rpts:
            self._route(r, t)

    def _route(self, r, req_type):
        if "venue_id" in r:
            self.venue_id[r["cl_ord_id"]] = r["venue_id"]
        target = self.sessions.get(self.owner.get(r["cl_ord_id"]))
        if target is None or target.closed.is_set():
            return   # production: persist and deliver on next logon
        if r["exec_type"] == "CANCEL_REJECT":
            order = self.oms.orders.get(r["orig_cl_ord_id"])
            target.send(fix.MsgType.ORDER_CANCEL_REJECT, [
                (37, self.venue_id.get(r["orig_cl_ord_id"], "NONE")), (11, r["cl_ord_id"]),
                (41, r["orig_cl_ord_id"]),
                (39, order.status if order else fix.OrdStatus.REJECTED),
                (434, "2" if req_type == fix.MsgType.ORDER_CANCEL_REPLACE_REQUEST else "1"),
                (58, r["text"])])
        else:
            raw = self.oms.to_fix(dict(r, venue_id=self.venue_id.get(r["cl_ord_id"], "0")))
            target.send(fix.MsgType.EXECUTION_REPORT, body_of(raw))


class Initiator(Session):
    """The client side: connect, log on, send orders, await execution reports."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.reports = asyncio.Queue()
        self.on_app = lambda sess, msg: self.reports.put_nowait(self._view(msg))

    @staticmethod
    def _view(msg):
        """Inbound app message as {tag name: value}, plus the PossDup flag."""
        d = {fix.NAME_BY_TAG.get(t, t): v for t, v in msg["pairs"]}
        d["poss_dup"] = msg["fields"].get(43) == "Y"
        return d

    @classmethod
    async def connect(cls, host, port, sender, target, hb=30, reset=True, verbose=False):
        reader, writer = await asyncio.open_connection(host, port)
        s = cls(sender, target, reader, writer, hb=hb, verbose=verbose)
        s.start()
        s.send(LOGON, [(98, 0), (108, hb)] + ([(141, "Y")] if reset else []))
        await asyncio.wait_for(s.logged_on.wait(), 5)
        return s

    def send_order(self, cl_ord_id, side, qty, price=None, symbol="TEST"):
        ord_type = fix.OrdType.MARKET if price is None else fix.OrdType.LIMIT
        return self.send(fix.MsgType.NEW_ORDER_SINGLE, body_of(
            fix.new_order_single(cl_ord_id, symbol, side, qty, ord_type=ord_type, price=price)))

    def cancel(self, cl_ord_id, orig_cl_ord_id, side, qty, symbol="TEST"):
        return self.send(fix.MsgType.ORDER_CANCEL_REQUEST, body_of(
            fix.cancel_request(cl_ord_id, orig_cl_ord_id, symbol, side, qty)))

    def replace(self, cl_ord_id, orig_cl_ord_id, side, qty, price, symbol="TEST"):
        return self.send(fix.MsgType.ORDER_CANCEL_REPLACE_REQUEST, body_of(
            fix.cancel_replace_request(cl_ord_id, orig_cl_ord_id, symbol, side, qty, price)))

    async def next_report(self, timeout=2.0):
        return await asyncio.wait_for(self.reports.get(), timeout)


# ---------------------------------------------------------------------------
async def _until(cond, timeout=3.0, what="condition"):
    end = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > end:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(0.01)


def _types(sess, t, tags=None):
    tags = tags or {}
    return [m for m in sess.received if m["fields"].get(35) == t
            and all(m["fields"].get(k) == v for k, v in tags.items())]


async def _demo():
    t0 = time.monotonic()
    venue = Acceptor(OMS(symbol="TEST"), clients={"CLIENT1", "CLIENT2", "CLIENT3"})
    port = await venue.start()

    # (1) logon handshake: both sides start at 1, Logon consumes seq 1 each way
    cli = await Initiator.connect("127.0.0.1", port, "CLIENT1", "VENUE", hb=1)
    acc = venue.sessions["CLIENT1"]
    assert (cli.out_seq, cli.in_seq, acc.out_seq, acc.in_seq) == (2, 2, 2, 2)
    assert acc.hb == 1, "acceptor adopts the initiator's HeartBtInt"
    logon = _types(cli, LOGON)[0]["fields"]
    assert logon[34] == "1" and logon[108] == "1" and logon[98] == "0" and logon[141] == "Y"
    print(f"  (1) logon ok: seqs out/in = {cli.out_seq}/{cli.in_seq}")

    # background: a client that goes completely silent must be disconnected
    hung = await Initiator.connect("127.0.0.1", port, "CLIENT2", "VENUE", hb=1)
    hung.silent = True
    hung_acc = venue.sessions["CLIENT2"]

    # refused logon: wrong TargetCompID gets a disconnect, no session
    r, w = await asyncio.open_connection("127.0.0.1", port)
    w.write(fix.encode(LOGON, [(52, utc_now()), (98, 0), (108, 1)], "CLIENT1", "NOT_US", 1).encode())
    assert await asyncio.wait_for(r.read(), 2) == b"", "bad CompID must be disconnected"
    w.close()

    # seq too low without PossDup is a serious error: Logout with reason
    low = await Initiator.connect("127.0.0.1", port, "CLIENT3", "VENUE", hb=1)
    low._write(fix.encode(HEARTBEAT, [(52, utc_now())], "CLIENT3", "VENUE", 1))
    await asyncio.wait_for(low.closed.wait(), 2)
    assert "too low" in _types(low, LOGOUT)[0]["fields"][58]
    print("  (1) bad CompID refused; MsgSeqNum too low -> Logout:",
          _types(low, LOGOUT)[0]["fields"][58])

    # (2) order round trip through the real OMS + book
    cli.send_order("B1", fix.Side.BUY, 50, price=10_005)
    cli.send_order("B2", fix.Side.BUY, 50, price=10_000)
    cli.send_order("S1", fix.Side.SELL, 80, price=10_000)   # sweeps B1 then 30 of B2
    reps = [await cli.next_report() for _ in range(7)]
    by = {}
    for rp in reps:
        by.setdefault(rp["ClOrdID"], []).append(rp)
    assert [x["ExecType"] for x in by["B1"]] == [fix.ExecType.NEW, fix.ExecType.TRADE]
    s1 = by["S1"][-1]
    assert s1["OrdStatus"] == fix.OrdStatus.FILLED and s1["CumQty"] == "80"
    assert abs(float(s1["AvgPx"]) - (50 * 10_005 + 30 * 10_000) / 80) < 1e-6, s1["AvgPx"]
    b2 = by["B2"][-1]
    assert b2["OrdStatus"] == fix.OrdStatus.PARTIALLY_FILLED
    assert (b2["CumQty"], b2["LeavesQty"], b2["LastQty"]) == ("30", "20", "30")
    assert all(not rp["poss_dup"] and rp["OrderID"] != "0" for rp in reps)
    print(f"  (2) round trip ok: S1 filled 80 @ AvgPx {s1['AvgPx']}, "
          f"B2 partial {b2['CumQty']}/50")

    # (3)+(4) gap recovery: the venue loses the Canceled report on the way out,
    # then sends a heartbeat. The client sees seq jump, asks for a resend, and
    # gets the report back flagged PossDup, with the heartbeat gap-filled.
    acc.drop_next_app = 1
    cli.cancel("C1", "B2", fix.Side.BUY, 50)
    await _until(lambda: acc.drop_next_app == 0, what="drop")
    dropped = acc.out_seq - 1
    acc.send(HEARTBEAT)
    rp = await cli.next_report()
    assert rp["ExecType"] == fix.ExecType.CANCELED and rp["ClOrdID"] == "B2"
    assert rp["poss_dup"] and rp["MsgSeqNum"] == str(dropped) and 122 in rp
    assert (RESEND_REQUEST, [(7, dropped), (16, 0)]) in [s[:2] for s in cli.store.values()]
    await _until(lambda: _types(cli, SEQUENCE_RESET, {123: "Y"}), what="gap fill")
    gf = _types(cli, SEQUENCE_RESET, {123: "Y"})[0]["fields"]
    assert gf[34] == str(dropped + 1) and int(gf[36]) == dropped + 2 and gf[43] == "Y"
    await _until(lambda: cli.in_seq == acc.out_seq and not cli.queue, what="resync")
    print(f"  (3) gap at seq {dropped} detected, ResendRequest(7={dropped},16=0) sent, "
          f"report replayed with PossDupFlag=Y")
    print(f"  (4) heartbeat at seq {dropped + 1} replaced by GapFill NewSeqNo={gf[36]}")

    # (4) again, wholesale: replay everything from 1. Logon and heartbeats must
    # come back as GapFills, business messages as PossDup copies, and none of
    # them may reach the client app a second time.
    before = len(cli.received)
    cli.send(RESEND_REQUEST, [(7, 1), (16, 0)])
    await _until(lambda: any(m["fields"].get(34) == "1" for m in cli.received[before:]),
                 what="full replay")
    await asyncio.sleep(0.05)
    replay = cli.received[before:]
    first = next(m["fields"] for m in replay if m["fields"][34] == "1")
    assert first[35] == SEQUENCE_RESET and first[123] == "Y" and first[36] == "2", first
    assert not any(m["fields"][35] in (LOGON, HEARTBEAT) and m["fields"].get(43) == "Y"
                   for m in replay), "admin messages must not be replayed"
    apps = [m for m in replay if m["fields"][35] == fix.MsgType.EXECUTION_REPORT]
    assert len(apps) == 8 and all(m["fields"][43] == "Y" for m in apps)
    assert cli.reports.empty(), "duplicates must not be re-delivered"
    print(f"  (4) full replay: {len(apps)} reports as PossDup, "
          f"{len(replay) - len(apps)} GapFills, 0 duplicates delivered")

    # (6) bad checksum: discarded, seq not consumed, session survives
    bad = fix.encode(TEST_REQUEST, [(52, utc_now()), (112, "BAD")], "CLIENT1", "VENUE", cli.out_seq)
    bad = bad[:-4] + f"{(int(bad[-4:-1]) + 1) % 256:03d}" + fix.SOH
    cli.writer.write(b"noise" + bad.encode())
    cli.send(TEST_REQUEST, [(112, "ALIVE")])
    await _until(lambda: _types(cli, HEARTBEAT, {112: "ALIVE"}), what="ALIVE heartbeat")
    assert acc.garbled == 1 and not acc.closed.is_set() and not acc.resend_pending
    assert any("checksum" in line for line in acc.log)
    print("  (6) bad checksum discarded, session still up:",
          next(l for l in acc.log if "checksum" in l))

    # (5) silence: the client stops heartbeating. The venue sends its own idle
    # heartbeat, then a TestRequest, which the client answers by echoing 112.
    cli.send_heartbeats = False
    n_hb = len(_types(cli, HEARTBEAT))
    await _until(lambda: _types(cli, TEST_REQUEST), timeout=3, what="TestRequest")
    await _until(lambda: acc.test_req_id is None, what="TestRequest answer")
    assert len(_types(cli, HEARTBEAT)) > n_hb, "idle side must heartbeat"
    tr = _types(cli, TEST_REQUEST)[0]["fields"][112]
    assert _types(acc, HEARTBEAT, {112: tr}), "TestReqID must be echoed"
    assert not acc.closed.is_set()
    cli.send_heartbeats = True
    print(f"  (5) idle heartbeat sent; TestRequest {tr} answered with 112={tr}")

    # (5) and the hung client, silent all along, has been cut off
    await asyncio.wait_for(hung_acc.closed.wait(), 3)
    assert hung_acc.close_reason == "no response to TestRequest", hung_acc.close_reason
    assert _types(hung, TEST_REQUEST)
    print(f"  (5) silent peer disconnected: {hung_acc.close_reason}")

    # (7) clean logout: 5 out, 5 back, both sides closed
    await cli.logout("done for the day")
    await asyncio.wait_for(acc.closed.wait(), 2)
    assert _types(cli, LOGOUT) and _types(acc, LOGOUT, {58: "done for the day"})
    assert cli.close_reason.startswith("logout") and acc.close_reason.startswith("logout")
    print(f"  (7) logout ok: client '{cli.close_reason}', venue '{acc.close_reason}'")

    await hung.closed.wait()
    await venue.stop()
    return time.monotonic() - t0


def demo():
    elapsed = asyncio.run(_demo())
    print(f"ok — FIX session: logon, round trip, gap recovery, gap fill, heartbeats, "
          f"garbled discard, logout ({elapsed:.1f}s)")


if __name__ == "__main__":
    demo()
