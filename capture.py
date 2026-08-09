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

Run `python capture.py` for the self-check (no network).
Run `python capture.py --live BTCUSDT` to record (needs `pip install websockets`).
"""

import gzip
import json
import time
from pathlib import Path

APPLIED, BUFFERED, STALE, GAP = "applied", "buffered", "stale", "gap"


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
    url = f"wss://stream.binance.com:9443/ws/{sym.lower()}@depth@100ms"
    snap_url = f"https://api.binance.com/api/v3/depth?symbol={sym}&limit=1000"
    out = f"{outdir}/{sym}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl.gz"

    stream, writer = DepthStream(), Writer(out)
    deadline = time.time() + seconds
    print(f"recording {sym} -> {out}")

    async with websockets.connect(url) as ws:
        # Snapshot AFTER the socket is open, so no events are missed in between.
        snap = json.loads(urllib.request.urlopen(snap_url, timeout=10).read())
        stream.apply_snapshot(snap)
        writer.write({"type": "snapshot", "ts": time.time(), **snap})

        while time.time() < deadline:
            ev = json.loads(await ws.recv())
            status = stream.apply_event(ev)
            writer.write({"type": "diff", "ts": time.time(), "status": status, **ev})
            if status == GAP:
                # Resync rather than keep a book we know is wrong.
                print("gap detected — resyncing")
                snap = json.loads(urllib.request.urlopen(snap_url, timeout=10).read())
                stream.apply_snapshot(snap)
                writer.write({"type": "snapshot", "ts": time.time(), **snap})

    writer.close()
    print(f"{writer.count} messages, {stream.gaps} gaps, "
          f"book {stream.best_bid()} / {stream.best_ask()}")
    return out


# ---------------------------------------------------------------------------
# Self-check — the sequencing rules, offline
# ---------------------------------------------------------------------------
def _ev(U, u, bids=(), asks=()):
    return {"U": U, "u": u, "b": [list(x) for x in bids], "a": [list(x) for x in asks]}


def demo():
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

    print("ok — sequencing, gap detection, book updates, writer all pass")


if __name__ == "__main__":
    import sys

    if "--live" in sys.argv:
        import asyncio
        i = sys.argv.index("--live")
        sym = sys.argv[i + 1] if len(sys.argv) > i + 1 else "BTCUSDT"
        secs = int(sys.argv[i + 2]) if len(sys.argv) > i + 2 else 60
        asyncio.run(record(sym, secs))
    else:
        demo()
