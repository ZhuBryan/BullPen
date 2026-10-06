"""
End-to-end test of the LIVE capture path against a fake Binance feed.

capture.py's sequencing state machine is unit-tested in isolation, but the
`record()` coroutine, the websocket plumbing, the combined depth+trade stream
routing, the gzip writer, and the gap-triggered resync had never actually run.
Those are exactly the parts that break in production and that a pure unit test
cannot reach.

So this stands up a real websocket server and a real HTTP server on localhost
that speak Binance's wire format, points `record()` at them, and then asserts on
the file that comes out the other end. No network, no API key, deterministic.

The feed deliberately injects a SEQUENCE GAP partway through, which is the one
failure mode that silently corrupts a local book. The test asserts that the gap
is detected and that a resync snapshot follows it.

    pip install websockets
    python test_capture_live.py
"""

import asyncio
import gzip
import json
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import websockets

try:
    from engine import capture
except ImportError:
    import capture

SNAPSHOT = {
    "lastUpdateId": 100,
    "bids": [["99.50000000", "3.00000000"], ["99.40000000", "5.00000000"]],
    "asks": [["99.60000000", "2.00000000"], ["99.70000000", "4.00000000"]],
}


class SnapshotHandler(BaseHTTPRequestHandler):
    """Stands in for GET /api/v3/depth."""

    def do_GET(self):
        body = json.dumps(SNAPSHOT).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass        # keep the test output clean


def diff(U, u, bids=(), asks=()):
    return {"stream": "btcusdt@depth@100ms",
            "data": {"e": "depthUpdate", "U": U, "u": u,
                     "b": [list(x) for x in bids], "a": [list(x) for x in asks]}}


def trade(price, qty, maker):
    return {"stream": "btcusdt@trade",
            "data": {"e": "trade", "p": price, "q": qty, "m": maker, "T": 1}}


# The scripted feed. Note the jump from u=112 to U=200: a missing range, which
# is the corruption the resync logic exists to catch.
FEED = [
    diff(101, 105, bids=[("99.50000000", "4.00000000")]),
    trade("99.50000000", "1.00000000", True),
    diff(106, 112, asks=[("99.60000000", "0")]),
    diff(200, 205, bids=[("99.50000000", "9.00000000")]),   # <-- gap
    diff(206, 210, bids=[("99.45000000", "1.00000000")]),
]


async def feed_handler(ws):
    for msg in FEED:
        await ws.send(json.dumps(msg))
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.4)      # hold the socket open until record() stops


async def main():
    http = HTTPServer(("127.0.0.1", 0), SnapshotHandler)
    threading.Thread(target=http.serve_forever, daemon=True).start()
    snap_port = http.server_address[1]

    async with websockets.serve(feed_handler, "127.0.0.1", 0) as server:
        ws_port = server.sockets[0].getsockname()[1]

        # Point the module at our local servers instead of Binance.
        orig_ws, orig_snap = capture.WS_URL, capture.SNAPSHOT_URL
        capture.WS_URL = f"ws://127.0.0.1:{ws_port}/stream?streams={{low}}@depth@100ms/{{low}}@trade"
        capture.SNAPSHOT_URL = f"http://127.0.0.1:{snap_port}/api/v3/depth?symbol={{sym}}&limit=1000"
        try:
            with tempfile.TemporaryDirectory() as d:
                out = await capture.record("BTCUSDT", seconds=1.0, outdir=d)
                rows = [json.loads(l) for l in gzip.open(out, "rt")]
        finally:
            capture.WS_URL, capture.SNAPSHOT_URL = orig_ws, orig_snap

    kinds = [r["type"] for r in rows]
    snaps = [r for r in rows if r["type"] == "snapshot"]
    diffs = [r for r in rows if r["type"] == "diff"]
    trades = [r for r in rows if r["type"] == "trade"]
    statuses = [r["status"] for r in diffs]

    assert kinds[0] == "snapshot", f"first record must be the snapshot, got {kinds[0]}"
    assert trades, "the trade stream was never routed to the writer"
    assert trades[0]["p"] == "99.50000000" and trades[0]["m"] is True

    # the scripted gap must be caught, not silently applied
    assert "gap" in statuses, f"sequence gap was not detected: {statuses}"
    gap_at = statuses.index("gap")
    assert statuses[:gap_at].count("applied") >= 2, "clean diffs should apply first"

    # and a fresh snapshot must follow it, or the local book stays wrong
    gap_row = diffs[gap_at]
    after = rows[rows.index(gap_row) + 1:]
    assert any(r["type"] == "snapshot" for r in after), \
        "no resync snapshot after the gap: the book would stay silently corrupt"
    assert len(snaps) >= 2, f"expected an initial + a resync snapshot, got {len(snaps)}"

    # a dropped socket must be survived and recorded, not fatal. Binance
    # recycles connections routinely, so this is a normal event.
    drops = [r for r in rows if r["type"] == "disconnect"]
    assert drops, "the server closed the socket and nothing was logged"
    assert drops[0]["error"], "disconnect record should name the exception type"

    # the snapshot that follows a gap says why it was taken
    assert any(s.get("reason") == "gap" for s in snaps), \
        "resync snapshot should be labelled with its cause"
    assert snaps[0].get("reason") == "initial"

    # every record is timestamped and the file round-trips through gzip
    assert all("ts" in r for r in rows), "records must carry a capture timestamp"
    assert Path(out).suffix == ".gz"

    print(f"ok — live capture: {len(rows)} records "
          f"({len(snaps)} snapshots, {len(diffs)} diffs, {len(trades)} trades), "
          f"gap detected and resynced")


if __name__ == "__main__":
    asyncio.run(main())
