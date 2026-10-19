"""
Live L2 order book capture (Binance public websocket, no API key needed).

The hard part is NOT reading a socket. It is staying correctly synchronized
with the exchange, which is a real distributed-systems problem:

  * The diff stream sends incremental updates, each stamped with a range of
    update ids [U, u].
  * A REST snapshot gives you the book at one lastUpdateId.
  * Those two race each other, so the documented handshake is:
      1. start buffering diff events immediately
      2. fetch the snapshot
      3. throw away buffered events entirely older than the snapshot (u <= id)
      4. the first event you apply must straddle the snapshot:
             U <= lastUpdateId + 1 <= u
      5. from then on every event must be contiguous: U == prev_u + 1
  * If step 5 ever fails you have MISSED data. The local book is now silently
    wrong, and the only correct response is to resync from a fresh snapshot.
    Silently continuing is how you end up with a backtest built on a lie.

DepthStream below is that state machine, and it is pure: no sockets, no
threads, no clock. That is what makes the gap logic testable offline, which is
where the bugs actually are.

KRAKEN (WebSocket v2) is the second source, in KrakenStream further down. It
solves the same problem with a different integrity mechanism, and the contrast
is worth understanding:

  * Binance stamps every diff with a sequence range. If a number is missing,
    Binance has told you that YOU MISSED A MESSAGE. You know before you apply
    anything, and you resync.
  * Kraken v2 book messages carry no sequence number at all. Instead every
    message carries a CRC32 checksum of what the top 10 levels of the book
    should look like AFTER the update. You apply the update, compute the same
    checksum over your local book, and compare. A mismatch means YOUR STATE IS
    WRONG, whatever the cause (a dropped message, a bad parse, a rounding bug,
    a truncation mistake). It catches more kinds of bug than a sequence number,
    but only after the fact, and only for the top 10 levels.

Both paths write the SAME on-disk record format, so replay.py and research.py
read either exchange's files without knowing which one produced them.

Run `python capture.py` for the self-check (no network).
Run `python capture.py --live BTCUSDT 300` to record Binance
(needs `pip install websockets`), or
`python capture.py --live kraken BTC/USD 3600` to record Kraken.
"""

import asyncio
import gzip
import json
import time
import zlib
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path

APPLIED, BUFFERED, STALE, GAP = "applied", "buffered", "stale", "gap"

# Module-level so a test can point record() at a local fake feed. The live
# socket path is the part a unit test cannot reach, and it is where the real
# failures live, so it needs to be testable without touching the network.
WS_URL = ("wss://stream.binance.com:9443/stream?streams="
          "{low}@depth@100ms/{low}@trade")
SNAPSHOT_URL = "https://api.binance.com/api/v3/depth?symbol={sym}&limit=1000"
KRAKEN_WS_URL = "wss://ws.kraken.com/v2"

# Reconnect backoff: BASE, 2*BASE, 4*BASE ... capped at 30s. Module-level so a
# test can shrink it instead of sleeping for real.
BACKOFF_BASE = 2.0


def _backoff(attempt):
    return min(BACKOFF_BASE * 2 ** (attempt - 1), 30)


class DepthStream:
    """Binance diff-depth sequencing + local book. Pure state machine."""

    def __init__(self):
        self.bids: dict[str, str] = {}   # price -> qty, both as exchange strings
        self.asks: dict[str, str] = {}
        self.last_id: int | None = None  # last applied update id
        self.buffer: list[dict] = []
        self.synced = False
        self.gaps = 0

    # ---- snapshot ------------------------------------------------------
    def apply_snapshot(self, snap):
        self.bids = {p: q for p, q in snap["bids"] if float(q) > 0}
        self.asks = {p: q for p, q in snap["asks"] if float(q) > 0}
        self.last_id = int(snap["lastUpdateId"])
        self.synced = False
        # Replay whatever arrived while the snapshot was in flight.
        pending, self.buffer = self.buffer, []
        for ev in pending:
            self.apply_event(ev)
        return self.last_id

    # ---- diff events ---------------------------------------------------
    def apply_event(self, ev):
        U, u = int(ev["U"]), int(ev["u"])

        if self.last_id is None:          # snapshot hasn't landed yet
            self.buffer.append(ev)
            return BUFFERED

        if u <= self.last_id:             # entirely before the snapshot
            return STALE

        if not self.synced:
            # First event must straddle the snapshot, else we are missing the
            # span between them and the snapshot is useless.
            if not (U <= self.last_id + 1 <= u):
                self.gaps += 1
                return GAP
            self.synced = True
        elif U != self.last_id + 1:       # hole in the sequence
            self.gaps += 1
            self.synced = False
            return GAP

        self._apply_side(self.bids, ev.get("b", []))
        self._apply_side(self.asks, ev.get("a", []))
        self.last_id = u
        return APPLIED

    @staticmethod
    def _apply_side(side, levels):
        for price, qty in levels:
            if float(qty) == 0:
                side.pop(price, None)     # qty 0 means delete the level
            else:
                side[price] = qty

    # ---- introspection -------------------------------------------------
    def best_bid(self):
        return max(self.bids, key=float) if self.bids else None

    def best_ask(self):
        return min(self.asks, key=float) if self.asks else None


class Writer:
    """Batched gzip JSONL. One file per run.

    ponytail: JSONL+gzip is stdlib and plenty for capture-side throughput.
    Convert to Parquet at the analysis step, where columnar reads actually pay
    off — adding pyarrow to the capture path buys nothing today.
    """

    def __init__(self, path, batch=500):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.fh = gzip.open(path, "at")
        self.batch, self.pending, self.count = batch, [], 0

    def write(self, obj):
        self.pending.append(json.dumps(obj, separators=(",", ":")))
        self.count += 1
        if len(self.pending) >= self.batch:
            self.flush()

    def flush(self):
        if self.pending:
            self.fh.write("\n".join(self.pending) + "\n")
            self.pending.clear()

    def close(self):
        self.flush()
        self.fh.close()


# ---------------------------------------------------------------------------
# Live capture (only part that touches the network)
# ---------------------------------------------------------------------------
async def record(symbol="BTCUSDT", seconds=60, outdir="data"):
    import urllib.request
    import websockets  # pip install websockets

    sym = symbol.upper()
    low = sym.lower()
    # Combined stream: depth diffs AND trades. Trades are not optional extra —
    # depth alone cannot tell you whether a level shrank because someone traded
    # (consumes the front of the queue) or someone cancelled (does not). Without
    # that distinction, queue-position fill modelling is guesswork.
    url = WS_URL.format(low=low)
    snap_url = SNAPSHOT_URL.format(sym=sym)
    out = f"{outdir}/{sym}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl.gz"

    stream, writer = DepthStream(), Writer(out)
    deadline = time.time() + seconds
    print(f"recording {sym} -> {out}")

    def resnapshot(reason):
        snap = json.loads(urllib.request.urlopen(snap_url, timeout=10).read())
        stream.apply_snapshot(snap)
        writer.write({"type": "snapshot", "ts": time.time(), "reason": reason, **snap})

    # Outer loop reconnects. Binance recycles connections (every 24h, and
    # whenever the network hiccups), so a dropped socket is a NORMAL event, not
    # an error: without this the capture service dies with an unhandled
    # ConnectionClosed and the recording simply stops.
    #
    # Every reconnect re-snapshots. That is mandatory, not belt-and-braces: while
    # the socket was down the book kept moving, so the local copy is stale and
    # the next diff will not line up with it.
    attempt = 0
    while time.time() < deadline:
        try:
            async with websockets.connect(url) as ws:
                attempt = 0
                # Snapshot AFTER the socket is open, so nothing is missed between.
                resnapshot("initial" if not stream.synced else "reconnect")

                while time.time() < deadline:
                    msg = json.loads(await ws.recv())
                    data = msg.get("data", msg)

                    if data.get("e") == "trade":
                        writer.write({"type": "trade", "ts": time.time(),
                                      "p": data["p"], "q": data["q"],
                                      # True = buyer is the maker (sell aggressor)
                                      "m": data["m"], "T": data["T"]})
                        continue

                    status = stream.apply_event(data)
                    writer.write({"type": "diff", "ts": time.time(),
                                  "status": status, **data})
                    if status == GAP:
                        # Resync rather than keep a book we know is wrong.
                        print("gap detected — resyncing")
                        resnapshot("gap")
        except asyncio.CancelledError:
            raise
        except Exception as exc:          # noqa: BLE001 - dropped socket, DNS, 5xx
            if time.time() >= deadline:
                break
            attempt += 1
            backoff = _backoff(attempt)
            print(f"connection lost ({type(exc).__name__}) — "
                  f"reconnecting in {backoff}s")
            writer.write({"type": "disconnect", "ts": time.time(),
                          "error": type(exc).__name__})
            await asyncio.sleep(min(backoff, max(0.0, deadline - time.time())))

    writer.close()
    print(f"{writer.count} messages, {stream.gaps} gaps, "
          f"book {stream.best_bid()} / {stream.best_ask()}")
    return out


# ---------------------------------------------------------------------------
# Kraken WebSocket v2: book + trades, checksum-verified
# ---------------------------------------------------------------------------
# Price and quantity precision per pair: the number of decimal places Kraken
# quotes. These matter for TWO reasons:
#   1. The checksum is computed over the decimal strings at exactly this
#      precision. JSON numbers lose trailing zeros (0.10000000 arrives as 0.1),
#      so without the table you cannot rebuild the string Kraken hashed.
#   2. Converting to integers ("ticks" for price, "lots" for size) is exact at
#      this precision, which keeps the local book free of float noise.
# Kraken's `instrument` channel publishes these live (`price_precision`,
# `qty_precision` per pair); they are hardcoded here to keep the capture path
# to one subscription. A wrong entry shows up immediately as a checksum
# mismatch on the very first snapshot, so it cannot fail silently.
KRAKEN_PRECISION = {
    # symbol: (price_precision, qty_precision)
    "BTC/USD": (1, 8),
    "ETH/USD": (2, 8),
}

MISMATCH, DROPPED = "checksum_mismatch", "dropped"


def to_units(value, precision):
    """Decimal string/number -> exact integer in units of 10**-precision.

    68000.1 at precision 1 -> 680001 ticks; 0.005 at precision 8 -> 500000
    lots. Goes through Decimal(str(...)) so a float like 0.1 never becomes
    0.09999999 and lands on the wrong tick.
    """
    d = value if isinstance(value, Decimal) else Decimal(str(value))
    return int((d.scaleb(precision)).to_integral_value(ROUND_HALF_EVEN))


def from_units(n, precision):
    """Integer units -> fixed-precision decimal string (680001, 1 -> "68000.1").

    Always the same number of decimals, so the same price is always the same
    dict key in replay's DepthStream, exactly like Binance's padded strings.
    """
    if precision == 0:
        return str(n)
    scale = 10 ** precision
    return f"{n // scale}.{n % scale:0{precision}d}"


def kraken_checksum_input(bids, asks):
    """The exact string Kraken CRC32s. bids/asks: [(price_units, qty_units)], best first.

    Kraken's recipe, per level: format price and qty at the pair's precision,
    remove the ".", strip leading zeros, concatenate price then qty. Top 10
    asks (lowest first) come first, then top 10 bids (highest first).

    In integer units that recipe collapses to str(int): "0.00500000" -> remove
    "." -> "000500000" -> strip zeros -> "500000" == str(500000 lots). That is
    the whole reason the book is kept in integers.
    """
    return ("".join(f"{p}{q}" for p, q in asks[:10])
            + "".join(f"{p}{q}" for p, q in bids[:10]))


def kraken_checksum(bids, asks):
    return zlib.crc32(kraken_checksum_input(bids, asks).encode())  # unsigned 32-bit


class KrakenStream:
    """Kraken v2 book + trades for one symbol. Pure: no sockets, no clock.

    `handle(msg, ts)` turns one parsed websocket message into capture records
    in the same format Binance capture writes, plus a flag saying "resubscribe
    now, the local book is wrong".

    Kraken has no sequence numbers, but replay.py's DepthStream expects
    contiguous U/u ids. We stamp our own: one local id per VERIFIED update.
    Since only verified updates are written as diffs, the ids are contiguous
    by construction and a Kraken file replays through the unchanged Binance
    reader. They are bookkeeping, not exchange data.
    """

    def __init__(self, symbol="BTC/USD", depth=10, precision=None):
        if precision is None:
            if symbol not in KRAKEN_PRECISION:
                raise ValueError(
                    f"no precision for {symbol!r}; add it to KRAKEN_PRECISION "
                    f"(Kraken's instrument channel lists price/qty precision)")
            precision = KRAKEN_PRECISION[symbol]
        self.symbol, self.depth = symbol, depth
        self.pp, self.qp = precision
        self.bids: dict[int, int] = {}   # price ticks -> qty lots
        self.asks: dict[int, int] = {}
        self.synced = False
        self.seq = 0                     # local id of the last verified update
        self.next_reason = "initial"     # why the next snapshot is being taken
        self.verified = self.mismatches = self.dropped = self.snapshots = 0
        self.trades = 0

    # ---- book maths ----------------------------------------------------
    def _levels(self, levels):
        return [(to_units(l["price"], self.pp), to_units(l["qty"], self.qp))
                for l in levels]

    def top(self, n=10):
        bids = sorted(self.bids.items(), key=lambda kv: -kv[0])[:n]
        asks = sorted(self.asks.items())[:n]
        return bids, asks

    def checksum(self):
        return kraken_checksum(*self.top(10))

    def _truncate(self):
        """Keep only `depth` levels per side; return what fell off.

        Kraken does not send a delete for a level pushed out of your
        subscribed depth by a better-priced insert: the client is expected to
        drop it. Forget, and level 11 lingers and the checksum (and your
        depth features) go wrong.
        """
        removed = {}
        # worst bid = lowest price; worst ask = highest price
        for name, side, highest_first in (("b", self.bids, False), ("a", self.asks, True)):
            extra = sorted(side, reverse=highest_first)[:max(0, len(side) - self.depth)]
            for p in extra:
                del side[p]
            removed[name] = extra
        return removed["b"], removed["a"]

    def _apply(self, side, levels):
        for p, q in levels:
            if q == 0:
                side.pop(p, None)          # qty 0 deletes the level
            else:
                side[p] = q

    def apply_snapshot(self, entry):
        self.bids = dict(self._levels(entry.get("bids", [])))
        self.asks = dict(self._levels(entry.get("asks", [])))
        self._truncate()
        self.synced = self.checksum() == int(entry["checksum"])
        if not self.synced:
            self.mismatches += 1
            return MISMATCH
        self.verified += 1
        self.snapshots += 1
        return APPLIED

    def apply_update(self, entry):
        """Returns (status, bids_applied, asks_applied, bids_cut, asks_cut)."""
        if not self.synced:
            # Waiting for a snapshot (startup, reconnect, or after a mismatch).
            # Updates now are relative to a book we do not have: ignore them.
            self.dropped += 1
            return DROPPED, [], [], [], []
        b, a = self._levels(entry.get("bids", [])), self._levels(entry.get("asks", []))
        self._apply(self.bids, b)
        self._apply(self.asks, a)
        cut_b, cut_a = self._truncate()
        if self.checksum() != int(entry["checksum"]):
            # The book is wrong. We cannot tell which level, or why, so the only
            # honest move is to throw it away and ask for a fresh snapshot.
            self.synced = False
            self.mismatches += 1
            return MISMATCH, b, a, cut_b, cut_a
        self.verified += 1
        return APPLIED, b, a, cut_b, cut_a

    # ---- message -> records ---------------------------------------------
    def _fmt(self, levels):
        return [[from_units(p, self.pp), from_units(q, self.qp)] for p, q in levels]

    def snapshot_row(self, ts, reason, checksum):
        bids, asks = self.top(self.depth)
        return {"type": "snapshot", "ts": ts, "reason": reason, "exchange": "kraken",
                "symbol": self.symbol, "lastUpdateId": self.seq,
                "bids": self._fmt(bids), "asks": self._fmt(asks),
                "checksum": checksum, "price_precision": self.pp,
                "qty_precision": self.qp}

    def handle(self, msg, ts):
        """One parsed message -> (records to write, resubscribe_needed)."""
        rows, resub = [], False
        channel, kind = msg.get("channel"), msg.get("type")

        if "method" in msg:                              # subscribe/unsubscribe ack
            if not msg.get("success", True):
                rows.append({"type": "error", "ts": ts, "exchange": "kraken",
                             "method": msg["method"], "error": msg.get("error")})
            return rows, resub

        if channel == "book":
            for entry in msg.get("data", []):
                if entry.get("symbol") != self.symbol:
                    continue
                if kind == "snapshot":
                    status = self.apply_snapshot(entry)
                    if status == APPLIED:
                        rows.append(self.snapshot_row(ts, self.next_reason,
                                                      int(entry["checksum"])))
                        self.next_reason = "resubscribe"
                        continue
                    where, b, a, cb, ca = "snapshot", [], [], [], []
                else:
                    status, b, a, cb, ca = self.apply_update(entry)
                    if status == DROPPED:
                        continue
                    if status == APPLIED:
                        self.seq += 1
                        # Truncated levels are written as explicit qty-0 deletes
                        # so a reader that knows nothing about Kraken's depth
                        # rule (replay.py) rebuilds the identical book.
                        rows.append({
                            "type": "diff", "ts": ts, "status": APPLIED,
                            "exchange": "kraken", "U": self.seq, "u": self.seq,
                            "b": self._fmt(b + [(p, 0) for p in cb]),
                            "a": self._fmt(a + [(p, 0) for p in ca]),
                            "checksum": int(entry["checksum"]),
                            "exch_ts": entry.get("timestamp")})
                        continue
                    where = "update"
                # Mismatch. NOT written as a diff, so replay never applies it:
                # the replayed book holds the last verified state until the
                # fresh snapshot replaces it.
                rows.append({"type": MISMATCH, "ts": ts, "exchange": "kraken",
                             "where": where, "expected": int(entry["checksum"]),
                             "got": self.checksum(),
                             "b": self._fmt(b), "a": self._fmt(a)})
                self.next_reason = "checksum"
                resub = True

        elif channel == "trade" and kind == "update":
            # kind "snapshot" would be RECENT HISTORY, not live prints; replaying
            # those as if they just happened would invent queue fills.
            for t in msg.get("data", []):
                if t.get("symbol") != self.symbol:
                    continue
                self.trades += 1
                rows.append({
                    "type": "trade", "ts": ts, "exchange": "kraken",
                    "p": from_units(to_units(t["price"], self.pp), self.pp),
                    "q": from_units(to_units(t["qty"], self.qp), self.qp),
                    # Kraken names the AGGRESSOR side. A "sell" aggressor hit a
                    # resting bid, so the buyer was the maker: Binance's m=True.
                    "m": t.get("side") == "sell",
                    "T": _iso_ms(t.get("timestamp")),
                    "side": t.get("side"), "trade_id": t.get("trade_id")})
        # status / heartbeat messages carry no market data
        return rows, resub

    def reset(self, reason):
        """Forget sync (socket dropped): next snapshot is labelled `reason`."""
        self.synced = False
        self.next_reason = reason

    def best_bid(self):
        return from_units(max(self.bids), self.pp) if self.bids else None

    def best_ask(self):
        return from_units(min(self.asks), self.pp) if self.asks else None


def _iso_ms(stamp):
    """Kraken RFC3339 "2024-01-01T00:00:00.123456Z" -> epoch ms (Binance's T)."""
    if not stamp:
        return None
    from datetime import datetime, timezone
    try:
        # Python 3.10's fromisoformat wants exactly 3 or 6 fraction digits, so
        # parse the whole seconds and the fraction separately.
        whole, _, frac = str(stamp).rstrip("Z").partition(".")
        dt = datetime.fromisoformat(whole).replace(tzinfo=timezone.utc)
        return int(dt.timestamp()) * 1000 + int((frac + "000")[:3])
    except ValueError:
        return None


async def record_kraken(symbol="BTC/USD", seconds=60, outdir="data", depth=10):
    """Record Kraken v2 book + trades for `seconds` into a gzip JSONL file."""
    import websockets  # pip install websockets

    stream = KrakenStream(symbol, depth)
    out = (f"{outdir}/kraken_{symbol.replace('/', '')}_"
           f"{time.strftime('%Y%m%d_%H%M%S')}.jsonl.gz")
    writer = Writer(out)
    deadline = time.time() + seconds
    print(f"recording kraken {symbol} (depth {depth}) -> {out}")

    book_sub = {"channel": "book", "symbol": [symbol], "depth": depth}
    # snapshot=False on trades: we only want prints from now on (see handle()).
    trade_sub = {"channel": "trade", "symbol": [symbol], "snapshot": False}

    def req(method, params):
        return json.dumps({"method": method, "params": params})

    # Same reconnect discipline as Binance: a dropped socket is normal, and
    # every new connection starts from a fresh snapshot because the book moved
    # while we were gone. On Kraken the snapshot arrives on the socket as the
    # first book message after subscribing, so there is no REST race.
    attempt = 0
    while time.time() < deadline:
        try:
            async with websockets.connect(KRAKEN_WS_URL) as ws:
                attempt = 0
                stream.reset("reconnect" if stream.snapshots else "initial")
                await ws.send(req("subscribe", book_sub))
                await ws.send(req("subscribe", trade_sub))

                while time.time() < deadline:
                    try:
                        raw = await asyncio.wait_for(
                            ws.recv(), timeout=max(0.01, deadline - time.time()))
                    except asyncio.TimeoutError:
                        break
                    # parse_float=Decimal: never let a price pass through float.
                    msg = json.loads(raw, parse_float=Decimal)
                    rows, resub = stream.handle(msg, time.time())
                    for r in rows:
                        writer.write(r)
                        if r["type"] == "error":
                            print(f"kraken error: {r['error']}")
                    if resub:
                        # Kraken's resync: drop the subscription and take it
                        # again, which makes the server send a fresh snapshot.
                        hint = (" (on a snapshot: check KRAKEN_PRECISION)"
                                if rows and rows[-1].get("where") == "snapshot" else "")
                        print(f"checksum mismatch, resubscribing{hint}")
                        await ws.send(req("unsubscribe", book_sub))
                        await ws.send(req("subscribe", book_sub))
        except asyncio.CancelledError:
            raise
        except Exception as exc:          # noqa: BLE001 - dropped socket, DNS, 5xx
            if time.time() >= deadline:
                break
            attempt += 1
            backoff = _backoff(attempt)
            print(f"connection lost ({type(exc).__name__}), reconnecting in {backoff}s")
            writer.write({"type": "disconnect", "ts": time.time(),
                          "exchange": "kraken", "error": type(exc).__name__})
            await asyncio.sleep(min(backoff, max(0.0, deadline - time.time())))

    writer.close()
    print(f"{writer.count} records, {stream.verified} checksums verified, "
          f"{stream.mismatches} mismatches, {stream.trades} trades, "
          f"book {stream.best_bid()} / {stream.best_ask()}")
    return out


# ---------------------------------------------------------------------------
# Self-check — the sequencing rules, offline
# ---------------------------------------------------------------------------
def _ev(U, u, bids=(), asks=()):
    return {"U": U, "u": u, "b": [list(x) for x in bids], "a": [list(x) for x in asks]}


# Kraken's own published example (docs.kraken.com, "Book checksum (WebSocket
# v2)"): this snapshot must hash to 3310070434. Pinning the implementation to
# the exchange's test vector, not just to one we built by hand, is what makes
# the first live recording trustworthy.
_KRAKEN_DOC_BIDS = [("45283.5", "0.10000000"), ("45283.4", "1.54582015"), ("45282.1", "0.10000000"),
                    ("45281.0", "0.10000000"), ("45280.3", "1.54592586"), ("45279.0", "0.07990000"),
                    ("45277.6", "0.03310103"), ("45277.5", "0.30000000"), ("45277.3", "1.54602737"),
                    ("45276.6", "0.15445238")]
_KRAKEN_DOC_ASKS = [("45285.2", "0.00100000"), ("45286.4", "1.54571953"), ("45286.6", "1.54571109"),
                    ("45289.6", "1.54560911"), ("45290.2", "0.15890660"), ("45291.8", "1.54553491"),
                    ("45294.7", "0.04454749"), ("45296.1", "0.35380000"), ("45297.5", "0.09945542"),
                    ("45299.5", "0.18772827")]


def _kraken_doc_vector():
    units = lambda levels: [(to_units(p, 1), to_units(q, 8)) for p, q in levels]
    got = kraken_checksum(units(_KRAKEN_DOC_BIDS), units(_KRAKEN_DOC_ASKS))
    assert got == 3310070434, f"Kraken documentation test vector: got {got}"


def demo():
    _kraken_doc_vector()
    snap = {"lastUpdateId": 100, "bids": [["10.0", "5"]], "asks": [["11.0", "5"]]}

    # events arriving before the snapshot are buffered, then replayed
    s = DepthStream()
    assert s.apply_event(_ev(101, 105)) == BUFFERED
    assert len(s.buffer) == 1
    s.apply_snapshot(snap)
    assert s.synced and s.last_id == 105, "buffered event should replay after snapshot"

    # events entirely older than the snapshot are dropped
    s = DepthStream()
    s.apply_snapshot(snap)
    assert s.apply_event(_ev(90, 99)) == STALE
    assert s.last_id == 100

    # first event must straddle lastUpdateId + 1
    s = DepthStream()
    s.apply_snapshot(snap)
    assert s.apply_event(_ev(150, 160)) == GAP, "non-straddling first event must gap"
    assert not s.synced
    assert s.apply_event(_ev(99, 101)) == APPLIED, "straddling event should sync"
    assert s.synced and s.last_id == 101

    # contiguous events apply, a hole is caught
    s = DepthStream()
    s.apply_snapshot(snap)
    s.apply_event(_ev(101, 105))
    assert s.apply_event(_ev(106, 110)) == APPLIED
    assert s.apply_event(_ev(112, 115)) == GAP, "missed 111 must be detected"
    assert s.gaps == 1 and not s.synced

    # book maths: update, insert, and qty-0 delete
    s = DepthStream()
    s.apply_snapshot(snap)
    s.apply_event(_ev(101, 102, bids=[("10.0", "7"), ("9.5", "3")], asks=[("11.0", "0")]))
    assert s.bids["10.0"] == "7", "level should update"
    assert s.bids["9.5"] == "3", "level should insert"
    assert "11.0" not in s.asks, "qty 0 must delete the level"
    assert s.best_bid() == "10.0" and s.best_ask() is None

    # best bid/ask compare numerically, not as strings ("9.5" < "10.0")
    s = DepthStream()
    s.apply_snapshot({"lastUpdateId": 1, "bids": [["9.5", "1"], ["10.0", "1"]], "asks": []})
    assert s.best_bid() == "10.0", "string sort would wrongly pick 9.5"

    # writer round-trips through gzip
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = f"{d}/t.jsonl.gz"
        w = Writer(p, batch=2)
        for i in range(5):
            w.write({"i": i})
        w.close()
        with gzip.open(p, "rt") as fh:
            rows = [json.loads(x) for x in fh]
        assert [r["i"] for r in rows] == [0, 1, 2, 3, 4], "writer lost messages"

    _kraken_demo()
    print("ok — sequencing, gap detection, book updates, writer, "
          "kraken checksum/truncation/resync all pass")


def _kraken_msg(kind, bids, asks, checksum):
    lv = lambda xs: [{"price": Decimal(p), "qty": Decimal(q)} for p, q in xs]
    return {"channel": "book", "type": kind,
            "data": [{"symbol": "BTC/USD", "bids": lv(bids), "asks": lv(asks),
                      "checksum": checksum}]}


def _kraken_demo():
    # exact unit conversion, including values JSON stripped of trailing zeros
    assert to_units("68000.1", 1) == 680001 and to_units(Decimal("0.005"), 8) == 500000
    assert to_units(0.1, 8) == 10000000, "float 0.1 must not drift a lot"
    assert from_units(680001, 1) == "68000.1" and from_units(500000, 8) == "0.00500000"

    # checksum string: asks low->high, then bids high->low, price then qty
    bids, asks = [(1000, 5), (999, 120)], [(1001, 300000)]
    assert kraken_checksum_input(bids, asks) == "1001300000" "10005" "999120"
    assert kraken_checksum(bids, asks) == zlib.crc32(b"100130000010005999120")

    s = KrakenStream("BTC/USD", depth=2)
    snap = [("100.0", "1"), ("99.9", "2")], [("100.1", "3"), ("100.2", "4")]
    s.apply_snapshot({"bids": [{"price": p, "qty": q} for p, q in snap[0]],
                      "asks": [{"price": p, "qty": q} for p, q in snap[1]],
                      "checksum": 0})
    assert not s.synced and s.mismatches == 1, "bad snapshot checksum must not sync"
    good = s.checksum()                       # now craft messages that DO match
    rows, resub = s.handle(_kraken_msg("snapshot", *snap, good), ts=0)
    assert not resub and rows[0]["type"] == "snapshot" and s.synced
    assert rows[0]["bids"][0] == ["100.0", "1.00000000"], rows[0]["bids"]

    # a better bid at depth 2 pushes 99.9 out WITHOUT an explicit delete
    s2 = KrakenStream("BTC/USD", depth=2)
    s2.bids, s2.asks, s2.synced = dict(s.bids), dict(s.asks), True
    s2._apply(s2.bids, [(1000 + 1, 5 * 10**8)]); s2._truncate()
    rows, resub = s.handle(_kraken_msg("update", [("100.1", "5")], [], s2.checksum()), 1)
    assert not resub and rows[0]["status"] == APPLIED, rows
    assert 999 not in s.bids, "level pushed beyond depth must be truncated"
    assert ["99.9", "0.00000000"] in rows[0]["b"], "truncation must be written as a delete"
    assert rows[0]["U"] == rows[0]["u"] == 1

    # wrong checksum -> mismatch row, resubscribe, then drop updates until snapshot
    rows, resub = s.handle(_kraken_msg("update", [("100.1", "6")], [], 12345), 2)
    assert resub and rows[0]["type"] == MISMATCH and not s.synced
    rows, resub = s.handle(_kraken_msg("update", [("100.1", "7")], [], 1), 3)
    assert rows == [] and not resub and s.dropped == 1, "unsynced update must be ignored"
    rows, _ = s.handle(_kraken_msg("snapshot", *snap, good), 4)
    assert rows[0]["reason"] == "checksum" and s.synced

    # checksum always covers the top 10 only, whatever the subscribed depth
    deep = KrakenStream("BTC/USD", depth=25)
    deep.bids = {1000 - i: 1 for i in range(25)}
    deep.asks = {1001 + i: 1 for i in range(25)}
    ten = KrakenStream("BTC/USD", depth=10)
    ten.bids = {1000 - i: 1 for i in range(10)}
    ten.asks = {1001 + i: 1 for i in range(10)}
    assert deep.checksum() == ten.checksum()

    # trades: Kraken names the aggressor; a sell aggressor means buyer was maker
    rows, _ = s.handle({"channel": "trade", "type": "update", "data": [
        {"symbol": "BTC/USD", "side": "sell", "price": Decimal("100.0"),
         "qty": Decimal("0.5"), "trade_id": 7, "timestamp": "2024-01-01T00:00:00.5Z"}]}, 5)
    assert rows[0]["m"] is True and rows[0]["q"] == "0.50000000"
    assert rows[0]["T"] == 1704067200500
    assert s.handle({"channel": "trade", "type": "snapshot", "data": [
        {"symbol": "BTC/USD", "side": "buy", "price": 1, "qty": 1}]}, 6)[0] == [], \
        "trade snapshot is history, not live prints"

    try:
        KrakenStream("DOGE/XYZ")
        raise AssertionError("unknown symbol must refuse rather than guess precision")
    except ValueError:
        pass


if __name__ == "__main__":
    import sys

    # python capture.py --live BTCUSDT 300            (Binance, as before)
    # python capture.py --live binance BTCUSDT 300
    # python capture.py --live kraken BTC/USD 3600 [depth]
    if "--live" in sys.argv:
        args = sys.argv[sys.argv.index("--live") + 1:]
        venue = args.pop(0).lower() if args and args[0].lower() in ("kraken", "binance") else "binance"
        if venue == "kraken":
            sym = args[0] if args else "BTC/USD"
            secs = float(args[1]) if len(args) > 1 else 60
            depth = int(args[2]) if len(args) > 2 else 10
            asyncio.run(record_kraken(sym, secs, depth=depth))
        else:
            sym = args[0] if args else "BTCUSDT"
            secs = int(args[1]) if len(args) > 1 else 60
            asyncio.run(record(sym, secs))
    else:
        demo()
