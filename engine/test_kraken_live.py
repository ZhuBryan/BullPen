"""
End-to-end test of the LIVE Kraken capture path against a fake Kraken v2 feed.

Same idea as test_capture_live.py: stand up a real websocket server on
localhost that speaks Kraken's WebSocket v2 wire format, point
`capture.record_kraken()` at it, and assert on the file that comes out.
No network, no API key, deterministic.

What the scripted feed does, per connection:

  connection 1
    status message, subscribe acks, a 10-level snapshot
    updates, including one better bid that pushes level 10 out of the book
      WITHOUT an explicit delete (the client must truncate)
    trades (one buy aggressor, one sell aggressor) and a heartbeat
    a CORRUPTED update: the server's true book changes one way, the message
      says another, and the checksum describes the true book. That is what a
      dropped or garbled message looks like from the client's side.
    waits for the client to unsubscribe + resubscribe, then sends a fresh
      snapshot, two more updates and a trade, and DROPS the socket
  connection 2
    a fresh snapshot (the reconnect), one update, then holds open

Every checksum the server sends is computed here, independently of
capture.py, with plain string formatting. The test then asserts the client
verified them, caught the bad one, resynced, survived the drop, captured the
trades, and that the file replays through replay.replay into the exact book
the server holds.

    pip install websockets
    python test_kraken_live.py
"""

import asyncio
import gzip
import json
import tempfile
import zlib
from decimal import Decimal

import websockets

try:
    from engine import capture, replay as replay_mod
except ImportError:
    import capture
    import replay as replay_mod

SYMBOL, DEPTH, PP, QP = "BTC/USD", 10, 1, 8      # BTC/USD: 1 price dp, 8 qty dp


# ---------------------------------------------------------------------------
# Independent checksum: string formatting, no integers, no capture.py code
# ---------------------------------------------------------------------------
def _fmt(x, prec):
    """Kraken's per-field rule: format at precision, drop ".", strip leading 0s."""
    return f"{float(x):.{prec}f}".replace(".", "").lstrip("0")


def expected_input(bids, asks):
    """bids/asks: {price: qty}. Top 10 asks ascending, then top 10 bids descending."""
    a = sorted(asks, key=float)[:10]
    b = sorted(bids, key=float, reverse=True)[:10]
    return ("".join(_fmt(p, PP) + _fmt(asks[p], QP) for p in a)
            + "".join(_fmt(p, PP) + _fmt(bids[p], QP) for p in b))


def expected_checksum(bids, asks):
    return zlib.crc32(expected_input(bids, asks).encode())


def test_checksum_worked_example():
    """A checksum built by hand, so the recipe itself is pinned down.

    Book (BTC/USD, price precision 1, qty precision 8), as Kraken's JSON
    would carry it, i.e. with trailing zeros already lost:
        asks  50000.5 x 0.1    50001 x 2     50010.2 x 0.00012
        bids  50000.0 x 1.25   49999.9 x 0.5

    Field by field (format at precision, remove ".", strip leading zeros):
        ask 50000.5 -> "50000.5"  -> "500005"  qty 0.1     -> "0.10000000" -> "10000000"
        ask 50001   -> "50001.0"  -> "500010"  qty 2       -> "2.00000000" -> "200000000"
        ask 50010.2 -> "50010.2"  -> "500102"  qty 0.00012 -> "0.00012000" -> "12000"
        bid 50000.0 -> "50000.0"  -> "500000"  qty 1.25    -> "1.25000000" -> "125000000"
        bid 49999.9 -> "49999.9"  -> "499999"  qty 0.5     -> "0.50000000" -> "50000000"

    CRC32 input, asks (ascending) then bids (descending):
        "500005100000005000102000000005001021200050000012500000049999950000000"
    """
    worked = "500005100000005000102000000005001021200050000012500000049999950000000"
    raw = ('{"channel":"book","type":"snapshot","data":[{"symbol":"BTC/USD",'
           '"bids":[{"price":50000.0,"qty":1.25},{"price":49999.9,"qty":0.5}],'
           '"asks":[{"price":50000.5,"qty":0.1},{"price":50001,"qty":2},'
           '{"price":50010.2,"qty":0.00012}],"checksum":%d}]}'
           % zlib.crc32(worked.encode()))
    msg = json.loads(raw, parse_float=Decimal)

    s = capture.KrakenStream(SYMBOL, DEPTH)
    rows, resub = s.handle(msg, ts=0.0)
    assert capture.kraken_checksum_input(*s.top(10)) == worked, \
        capture.kraken_checksum_input(*s.top(10))
    assert s.checksum() == zlib.crc32(worked.encode())
    assert not resub and rows[0]["type"] == "snapshot", "worked example must verify"
    # and the test's own independent implementation agrees with the hand string
    assert expected_input({"50000.0": 1.25, "49999.9": 0.5},
                          {"50000.5": 0.1, "50001": 2, "50010.2": 0.00012}) == worked


# ---------------------------------------------------------------------------
# The fake server's true book and wire messages
# ---------------------------------------------------------------------------
class TrueBook:
    """What the exchange actually holds. Prices/qtys as floats, like its JSON."""

    def __init__(self):
        self.bids = {50000.0 - i * 0.5: 1.0 + i * 0.25 for i in range(DEPTH)}
        self.asks = {50000.5 + i * 0.5: 0.1 + i * 0.1 for i in range(DEPTH)}
        self.sent_checksums = []          # checksums of every GOOD update sent

    def checksum(self):
        return expected_checksum(self.bids, self.asks)

    def apply(self, bids=(), asks=()):
        for side, levels in ((self.bids, bids), (self.asks, asks)):
            for p, q in levels:
                if q == 0:
                    side.pop(p, None)
                else:
                    side[p] = q
        # the exchange's view is also only `depth` deep
        for p in sorted(self.bids, reverse=True)[DEPTH:]:
            del self.bids[p]
        for p in sorted(self.asks)[DEPTH:]:
            del self.asks[p]

    def snapshot(self):
        lv = lambda side, rev: [{"price": p, "qty": side[p]}
                                for p in sorted(side, reverse=rev)]
        return {"channel": "book", "type": "snapshot",
                "data": [{"symbol": SYMBOL, "bids": lv(self.bids, True),
                          "asks": lv(self.asks, False),
                          "checksum": self.checksum()}]}

    def update(self, bids=(), asks=(), lie=None):
        """Apply to the true book; send it. `lie` replaces what is SENT."""
        self.apply(bids, asks)
        sent_b, sent_a = lie if lie else (bids, asks)
        if not lie:
            self.sent_checksums.append(self.checksum())
        lv = lambda xs: [{"price": p, "qty": q} for p, q in xs]
        return {"channel": "book", "type": "update",
                "data": [{"symbol": SYMBOL, "bids": lv(sent_b), "asks": lv(sent_a),
                          "checksum": self.checksum(),
                          "timestamp": "2026-10-19T12:00:00.123456Z"}]}


def trade(side, price, qty, tid):
    return {"channel": "trade", "type": "update",
            "data": [{"symbol": SYMBOL, "side": side, "price": price, "qty": qty,
                      "ord_type": "market", "trade_id": tid,
                      "timestamp": "2026-10-19T12:00:00.250000Z"}]}


def ack(req):
    return {"method": req["method"], "success": True,
            "result": {"channel": req["params"]["channel"], "symbol": SYMBOL}}


class FakeKraken:
    def __init__(self):
        self.book = TrueBook()
        self.connections = 0
        self.received = []                # every client request, in order

    async def recv_req(self, ws):
        req = json.loads(await ws.recv())
        self.received.append((self.connections, req["method"], req["params"]["channel"]))
        await ws.send(json.dumps(ack(req)))
        return req

    async def handler(self, ws):
        self.connections += 1
        b = self.book
        send = lambda m: ws.send(json.dumps(m))
        await send({"channel": "status", "type": "update",
                    "data": [{"system": "online", "api_version": "v2"}]})
        for _ in range(2):                # book + trade subscriptions
            await self.recv_req(ws)
        await send(b.snapshot())

        if self.connections == 1:
            script = [
                b.update(bids=[(50000.0, 3.0)]),                       # qty change
                trade("buy", 50000.5, 0.05, 1),
                # better bid at 50000.2: level 10 (49995.5) falls off, no delete sent
                b.update(bids=[(50000.2, 0.4)]),
                b.update(asks=[(50000.5, 0.0), (50005.5, 0.7)]),       # delete + refill
                {"channel": "heartbeat"},
                trade("sell", 50000.2, 0.4, 2),
                b.update(bids=[(50000.2, 0.0), (49995.5, 2.0)]),
            ]
            for m in script:
                await send(m)
                await asyncio.sleep(0.01)
            # the corrupted message: truth is 50001.0 -> 9.0, wire says 1.0
            await send(b.update(asks=[(50001.0, 9.0)], lie=([], [(50001.0, 1.0)])))
            # the client must now unsubscribe and resubscribe the book
            await self.recv_req(ws)
            await self.recv_req(ws)
            await send(b.snapshot())
            for m in [b.update(bids=[(49999.5, 0.0), (49995.0, 1.0)]),
                      b.update(asks=[(50001.0, 8.5)]),
                      trade("buy", 50001.0, 0.5, 3)]:
                await send(m)
                await asyncio.sleep(0.01)
            return                        # handler returns -> socket dropped
        await send(b.update(bids=[(50000.0, 2.5)]))
        async for _ in ws:                # hold open until the client leaves
            pass


async def test_durability():
    """The four ways a real recording died or could die, each now handled.

    1. A socket that stays open but goes silent must be treated as dead and
       reconnected (Kraken heartbeats every second, so silence means trouble).
    2. Cancelling the recorder (what Ctrl+C does) must still close the file,
       leaving a complete gzip stream.
    3. A recording that's still running must be readable up to its last batch.
    4. A recording cut off mid-write must replay up to the cut, not crash.
    """
    import os
    import shutil
    from replay import _records

    connections = 0

    async def silent(ws):                # accepts, then never says a word
        nonlocal connections
        connections += 1
        await ws.wait_closed()           # until the client gives up on it

    async with websockets.serve(silent, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        saved = capture.KRAKEN_WS_URL, capture.BACKOFF_BASE, capture.SILENCE_TIMEOUT
        capture.KRAKEN_WS_URL = f"ws://127.0.0.1:{port}/v2"
        capture.BACKOFF_BASE, capture.SILENCE_TIMEOUT = 0.05, 0.3
        try:
            with tempfile.TemporaryDirectory() as d:
                # 1. silence -> reconnect, logged as a disconnect
                out = await capture.record_kraken(SYMBOL, seconds=1.6, outdir=d)
                rows = [json.loads(l) for l in gzip.open(out, "rt")]
                drops = [r for r in rows if r["type"] == "disconnect"]
                assert connections >= 3 and drops and drops[0]["error"] == "ConnectionError", \
                    (connections, drops)

                # 2. cancel mid-recording (Ctrl+C) -> the file is still closed
                task = asyncio.create_task(capture.record_kraken(SYMBOL, seconds=60, outdir=d))
                await asyncio.sleep(0.8)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                newest = max((os.path.join(d, f) for f in os.listdir(d)), key=os.path.getmtime)
                with gzip.open(newest, "rt") as fh:
                    fh.read()            # raises EOFError if the end marker is missing
        finally:
            capture.KRAKEN_WS_URL, capture.BACKOFF_BASE, capture.SILENCE_TIMEOUT = saved

    with tempfile.TemporaryDirectory() as d:
        # 3. readable while still open: each flushed batch is on disk
        path, peek = os.path.join(d, "live.jsonl.gz"), os.path.join(d, "peek.jsonl.gz")
        w = capture.Writer(path, batch=10)
        for i in range(25):
            w.write({"type": "trade", "i": i})
        shutil.copy(path, peek)          # what a reader sees mid-recording
        assert [r["i"] for r in _records(peek)] == list(range(20)), "flushed batches not on disk"
        w.close()
        assert [r["i"] for r in _records(path)] == list(range(25))

        # 4. cut off mid-write: everything before the cut survives
        with open(path, "rb") as fh:
            data = fh.read()
        with open(peek, "wb") as fh:
            fh.write(data[: len(data) * 2 // 3])
        got = [r["i"] for r in _records(peek)]
        assert got == list(range(len(got))), "records after a cut must not be garbled"

    print("ok — durability: silent feed reconnects, cancel closes the file, "
          "open recordings are readable, cut-off recordings replay up to the cut")


async def main():
    test_checksum_worked_example()
    await test_durability()

    fake = FakeKraken()
    async with websockets.serve(fake.handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        orig_url, orig_backoff = capture.KRAKEN_WS_URL, capture.BACKOFF_BASE
        capture.KRAKEN_WS_URL = f"ws://127.0.0.1:{port}/v2"
        capture.BACKOFF_BASE = 0.05      # reconnect fast; the logic is the same
        try:
            with tempfile.TemporaryDirectory() as d:
                out = await capture.record_kraken(SYMBOL, seconds=1.5, outdir=d)
                rows = [json.loads(l) for l in gzip.open(out, "rt")]
                replayed = list(_replay_checks(out))
        finally:
            capture.KRAKEN_WS_URL, capture.BACKOFF_BASE = orig_url, orig_backoff

    snaps = [r for r in rows if r["type"] == "snapshot"]
    diffs = [r for r in rows if r["type"] == "diff"]
    trades = [r for r in rows if r["type"] == "trade"]
    bad = [r for r in rows if r["type"] == capture.MISMATCH]
    drops = [r for r in rows if r["type"] == "disconnect"]

    # good messages verify: every diff written carries the checksum the server
    # computed independently, in order, and nothing good was rejected
    assert all(r["status"] == "applied" for r in diffs)
    assert [r["checksum"] for r in diffs] == fake.book.sent_checksums, \
        "client did not verify exactly the good updates the server sent"
    assert len(diffs) == 7, f"expected 4 + 2 + 1 verified updates, got {len(diffs)}"

    # truncation: the pushed-out level was written as an explicit delete
    assert ["49995.5", "0.00000000"] in diffs[1]["b"], diffs[1]["b"]

    # the corrupted update is caught, and not written as a diff
    assert len(bad) == 1, f"expected one checksum mismatch, got {len(bad)}"
    assert bad[0]["where"] == "update" and bad[0]["got"] != bad[0]["expected"]

    # ... and Kraken's resync happens: unsubscribe + subscribe on the SAME socket,
    # then a fresh snapshot labelled with its cause
    assert fake.received[2:4] == [(1, "unsubscribe", "book"), (1, "subscribe", "book")], \
        fake.received
    assert [s["reason"] for s in snaps] == ["initial", "checksum", "reconnect"], \
        [s["reason"] for s in snaps]
    assert rows.index(snaps[1]) == rows.index(bad[0]) + 1, \
        "the resync snapshot must be the next book record after the mismatch"

    # the dropped socket is survived: logged, reconnected, fresh snapshot
    assert drops and drops[0]["error"], "socket drop not recorded"
    assert fake.connections == 2, f"expected a reconnect, saw {fake.connections} connections"
    assert rows.index(drops[0]) < rows.index(snaps[2])
    # each written snapshot is itself the verified book: re-hash its levels
    for s in snaps:
        assert expected_checksum(dict(s["bids"]), dict(s["asks"])) == s["checksum"]

    # trades routed, with the aggressor mapped to Binance's buyer-is-maker flag
    assert [(t["p"], t["q"], t["m"]) for t in trades] == [
        ("50000.5", "0.05000000", False), ("50000.2", "0.40000000", True),
        ("50001.0", "0.50000000", False)], trades
    assert trades[0]["T"] == 1792411200250, trades[0]["T"]

    # replay: every replayed book state matches its checksum, and the final
    # replayed book is exactly the server's true book
    assert replayed, "replay produced nothing"
    final_bids, final_asks = replayed[-1]
    at_precision = lambda side: {round(p, PP): round(q, QP) for p, q in side.items()}
    assert at_precision(final_bids) == at_precision(fake.book.bids) and \
        at_precision(final_asks) == at_precision(fake.book.asks), \
        "replayed book differs from the exchange's book"

    print(f"ok — kraken live capture: {len(rows)} records ({len(snaps)} snapshots, "
          f"{len(diffs)} verified diffs, {len(trades)} trades), checksum mismatch "
          f"caught and resubscribed, dropped socket survived, replay matches")


def _replay_checks(path):
    """Replay the file; check each book state against its recorded checksum."""
    n = 0
    for kind, msg, book in replay_mod.replay(path):
        if kind != "book":
            continue
        bids = {float(p): float(q) for p, q in book.bids.items()}
        asks = {float(p): float(q) for p, q in book.asks.items()}
        assert expected_checksum(bids, asks) == msg["checksum"], \
            f"replayed book diverged at record {n} ({msg['type']})"
        n += 1
        yield bids, asks
    # replay's helpers work on the reconstructed book too
    assert replay_mod.best_pair(book) is not None


def demo():
    asyncio.run(main())


if __name__ == "__main__":
    demo()
