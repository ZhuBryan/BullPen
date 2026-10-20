"""
leadlag.py: does one exchange's price move before the other's?

Bitcoin trades on many venues at once. If arbitrageurs were instant, every
venue's price would move together. They aren't, so when news hits, the venue
where the informed traders act first moves first and the others follow a few
hundred milliseconds later. Measuring that lead is a classic market-structure
question, and the answer tells you where price discovery happens.

What this does:

  record   Kraken (WebSocket v2 `ticker`, event_trigger=bbo: a message on every
           best-bid/offer change) and Coinbase (Advanced Trade `ticker`, a
           message on every trade, with best bid/ask) on ONE event loop, so both
           are stamped by the same local clock. Coinbase's per-connection
           sequence numbers are checked for gaps. A venue silent for
           SILENCE_TIMEOUT is reconnected, and the file is always closed.

  analyse  Put both mid prices on a common time grid (last value carried
           forward), take returns, and correlate venue A's return now with
           venue B's return k steps later, for k from -L to +L. A peak at
           positive k means A moves first; the lag of the peak is the lead.

Caveat that matters: timestamps are when OUR machine received each message,
so the measured lead includes the difference in network delay from us to each
exchange. Recording from a datacenter (the GitHub Actions workflow does)
keeps that difference small and stable, but a lead under ~50 ms can't be
told apart from routing.

    python leadlag.py --live 3600                  record one hour
    python leadlag.py "data/leadlag_*.jsonl.gz"    analyse the newest recording
    python leadlag.py --study                      all analysed recordings
    python leadlag.py                              self-check
"""

import asyncio
import csv
import glob
import gzip
import json
import os
import sys
import time
from decimal import Decimal

import numpy as np

try:
    from engine import capture
    from engine.replay import _records
except ImportError:
    import capture
    from replay import _records

KRAKEN_URL = "wss://ws.kraken.com/v2"
COINBASE_URL = "wss://advanced-trade-ws.coinbase.com"   # public: no API key needed
GRID_MS, MAX_LAG_MS = 50, 2000
STUDY_FIELDS = ["file", "start", "kraken_updates", "coinbase_updates", "coinbase_gaps", "disconnects",
                "overlap_min", "peak_lag_ms", "peak_corr", "leader", "contemporaneous_corr"]


# ---------------------------------------------------------------------------
# Message parsing (pure: no sockets)
# ---------------------------------------------------------------------------
def parse_kraken(msg, symbol):
    """Kraken v2 ticker message -> [(bid, ask)] as strings, or []."""
    if msg.get("channel") != "ticker" or msg.get("type") not in ("snapshot", "update"):
        return []
    return [(str(d["bid"]), str(d["ask"])) for d in msg.get("data", [])
            if d.get("symbol") == symbol and d.get("bid") is not None and d.get("ask") is not None]


def parse_coinbase(msg, product):
    """Coinbase Advanced Trade ticker message -> ([(bid, ask)], sequence_num or None)."""
    if msg.get("channel") != "ticker":
        return [], msg.get("sequence_num")
    out = []
    for ev in msg.get("events", []):
        for t in ev.get("tickers", []):
            if t.get("product_id") == product and t.get("best_bid") and t.get("best_ask"):
                out.append((str(t["best_bid"]), str(t["best_ask"])))
    return out, msg.get("sequence_num")


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------
async def _venue(name, url, subscribe, parse, writer, deadline, stats):
    """One venue's connect / read / reconnect loop, writing BBO rows."""
    import websockets
    attempt = 0
    while time.time() < deadline:
        try:
            async with websockets.connect(url, max_size=2 ** 22) as ws:
                attempt, last_seq = 0, None
                for m in subscribe:
                    await ws.send(json.dumps(m))
                while time.time() < deadline:
                    left = deadline - time.time()
                    try:
                        raw = await asyncio.wait_for(
                            ws.recv(), timeout=max(0.01, min(capture.SILENCE_TIMEOUT, left)))
                    except asyncio.TimeoutError:
                        if time.time() >= deadline:
                            break
                        raise ConnectionError(f"{name} silent for {capture.SILENCE_TIMEOUT:.0f}s")
                    ts = time.time()
                    msg = json.loads(raw, parse_float=Decimal)
                    rows, seq = parse(msg)
                    # Coinbase numbers every message on a connection. A jump
                    # means we missed messages: count it, so a recording with
                    # holes is visible rather than silently smoothed over.
                    if seq is not None:
                        if last_seq is not None and seq != last_seq + 1:
                            stats[name + "_gaps"] += 1
                            writer.write({"v": name, "ts": ts, "event": "seq_gap",
                                          "expected": last_seq + 1, "got": seq})
                        last_seq = seq
                    for bid, ask in rows:
                        writer.write({"v": name, "ts": ts, "bid": bid, "ask": ask})
                        stats[name] += 1
        except asyncio.CancelledError:
            raise
        except Exception as exc:          # noqa: BLE001 - drops, DNS, 5xx, silence
            if time.time() >= deadline:
                break
            attempt += 1
            stats[name + "_reconnects"] += 1
            print(f"{name}: connection lost ({type(exc).__name__}), reconnecting")
            writer.write({"v": name, "ts": time.time(), "event": "disconnect",
                          "error": type(exc).__name__})
            await asyncio.sleep(min(capture._backoff(attempt), max(0.0, deadline - time.time())))


async def record(seconds=60, outdir="data", symbol="BTC/USD", product="BTC-USD"):
    """Record both venues' best bid/offer for `seconds` into one gzip JSONL file."""
    out = f"{outdir}/leadlag_{time.strftime('%Y%m%d_%H%M%S')}.jsonl.gz"
    writer = capture.Writer(out)
    deadline = time.time() + seconds
    stats = {k: 0 for k in ("kraken", "coinbase", "kraken_gaps", "coinbase_gaps",
                            "kraken_reconnects", "coinbase_reconnects")}
    print(f"recording kraken {symbol} + coinbase {product} best bid/offer -> {out}")
    kraken_sub = [{"method": "subscribe", "params": {
        "channel": "ticker", "symbol": [symbol], "event_trigger": "bbo"}}]
    # heartbeats keeps a quiet Coinbase connection open (it closes idle
    # subscriptions after 60-90 s); no JWT is required for public channels.
    coinbase_sub = [{"type": "subscribe", "channel": "ticker", "product_ids": [product]},
                    {"type": "subscribe", "channel": "heartbeats"}]
    try:
        await asyncio.gather(
            _venue("kraken", KRAKEN_URL, kraken_sub,
                   lambda m: (parse_kraken(m, symbol), None), writer, deadline, stats),
            _venue("coinbase", COINBASE_URL, coinbase_sub,
                   lambda m: parse_coinbase(m, product), writer, deadline, stats))
    finally:
        writer.close()
    print(f"kraken {stats['kraken']} updates ({stats['kraken_reconnects']} reconnects), "
          f"coinbase {stats['coinbase']} updates ({stats['coinbase_gaps']} sequence gaps, "
          f"{stats['coinbase_reconnects']} reconnects)")
    return out


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def load(path):
    """-> {venue: (ts array, mid array)} from a recording."""
    series = {"kraken": ([], []), "coinbase": ([], [])}
    for r in _records(path):
        if r.get("v") in series and "bid" in r:
            t, m = series[r["v"]]
            t.append(r["ts"])
            m.append((float(r["bid"]) + float(r["ask"])) / 2)
    return {v: (np.asarray(t, float), np.asarray(m, float)) for v, (t, m) in series.items()}


def _on_grid(ts, mid, grid):
    """Last value at or before each grid time (no peeking at later updates)."""
    i = np.searchsorted(ts, grid, side="right") - 1
    return mid[np.clip(i, 0, len(mid) - 1)], i >= 0


def lead_lag(a, b, grid_ms=GRID_MS, max_lag_ms=MAX_LAG_MS):
    """Cross-correlation of A's grid returns with B's, at lags -L..+L.

    corr[k] = corr(rA[t], rB[t + k]). Positive k with a high value means A's
    move now shows up in B k steps later: A leads.
    """
    (ta, ma), (tb, mb) = a, b
    t0, t1 = max(ta[0], tb[0]), min(ta[-1], tb[-1])
    if t1 - t0 < 60:
        raise ValueError("need at least a minute where both venues have data")
    grid = np.arange(t0, t1, grid_ms / 1000)
    pa, oka = _on_grid(ta, ma, grid)
    pb, okb = _on_grid(tb, mb, grid)
    ra, rb = np.diff(np.log(pa)), np.diff(np.log(pb))
    L = int(max_lag_ms / grid_ms)
    lags = np.arange(-L, L + 1)
    corr = []
    for k in lags:
        x, y = (ra[:len(ra) - k], rb[k:]) if k >= 0 else (ra[-k:], rb[:len(rb) + k])
        corr.append(float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else 0.0)
    corr = np.asarray(corr)
    return lags * grid_ms, corr, (t1 - t0) / 60


def analyse(path):
    s = load(path)
    lag_ms, corr, minutes = lead_lag(s["kraken"], s["coinbase"])
    k = int(np.argmax(corr))
    peak_lag = int(lag_ms[k])
    leader = "kraken" if peak_lag > 0 else "coinbase" if peak_lag < 0 else "tie"
    events = [r.get("event") for r in _records(path) if r.get("event")]
    m = os.path.basename(path)
    stamp = m.replace("leadlag_", "").replace(".jsonl.gz", "")
    return {
        "file": m, "start": stamp, "kraken_updates": len(s["kraken"][0]),
        "coinbase_updates": len(s["coinbase"][0]),
        "coinbase_gaps": events.count("seq_gap"), "disconnects": events.count("disconnect"),
        "overlap_min": round(minutes, 1), "peak_lag_ms": peak_lag,
        "peak_corr": round(float(corr[k]), 3), "leader": leader,
        "contemporaneous_corr": round(float(corr[len(corr) // 2]), 3),
        "_lags": lag_ms.tolist(), "_corr": corr.round(4).tolist(),
    }


def append_study(row, path="data/leadlag_study.csv"):
    rows = []
    if os.path.exists(path):
        with open(path, newline="") as fh:
            rows = [r for r in csv.DictReader(fh) if r["file"] != row["file"]]
    rows.append({k: row[k] for k in STUDY_FIELDS})
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=STUDY_FIELDS)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["start"]))


def report(r):
    print(f"{r['file']}: {r['overlap_min']} min overlap, kraken {r['kraken_updates']:,} "
          f"updates, coinbase {r['coinbase_updates']:,}")
    print("lag (ms, + = kraken first)   corr")
    for lag, c in zip(r["_lags"], r["_corr"]):
        if lag % 250 == 0 or lag == r["peak_lag_ms"]:
            mark = "  <- peak" if lag == r["peak_lag_ms"] else ""
            print(f"{lag:>8}                   {c:+.3f}{mark}")
    if r["leader"] == "tie":
        print("peak at zero lag: the venues move together at this resolution")
    else:
        print(f"{r['leader']} leads by about {abs(r['peak_lag_ms'])} ms "
              f"(corr {r['peak_corr']:+.3f} vs {r['contemporaneous_corr']:+.3f} at zero lag). "
              "Includes any difference in network delay to the two venues.")


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
def _synthetic(path, lead_ms, seed=0, independent=False, seconds=600):
    """Two venues observing one random walk; Kraken sees it lead_ms earlier."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 0.01)
    walk = 82000 * np.exp(np.cumsum(rng.normal(0, 2e-5, len(t))))
    other = 82000 * np.exp(np.cumsum(rng.normal(0, 2e-5, len(t)))) if independent else walk
    with gzip.open(path, "wt") as fh:
        for venue, series, delay, rate in (("kraken", walk, 0.0, 20), ("coinbase", other, lead_ms / 1000, 15)):
            times = np.sort(rng.uniform(0, seconds, int(seconds * rate)))
            for ts in times:
                i = max(0, int((ts - delay) / 0.01))      # what this venue knows at ts
                mid = series[min(i, len(series) - 1)]
                fh.write(json.dumps({"v": venue, "ts": 1.7e9 + ts, "bid": f"{mid - 0.05:.2f}",
                                     "ask": f"{mid + 0.05:.2f}"}) + "\n")


async def _fake_servers_test(tmp):
    import websockets
    cb_conns = {"n": 0}

    async def kraken(ws):
        sub = json.loads(await ws.recv())
        assert sub["params"]["event_trigger"] == "bbo" and sub["params"]["channel"] == "ticker"
        await ws.send(json.dumps({"method": "subscribe", "success": True}))
        for i in range(30):
            await ws.send(json.dumps({"channel": "heartbeat"}))
            await ws.send(json.dumps({"channel": "ticker", "type": "update", "data": [
                {"symbol": "BTC/USD", "bid": 82000.0 + i, "ask": 82000.1 + i}]}))
            await asyncio.sleep(0.02)
        await ws.wait_closed()

    async def coinbase(ws):
        cb_conns["n"] += 1
        subs = [json.loads(await ws.recv()) for _ in range(2)]
        assert {s["channel"] for s in subs} == {"ticker", "heartbeats"}
        if cb_conns["n"] > 1:
            await ws.wait_closed()       # second connection: just stay quiet
            return
        for seq in (1, 2, 3, 5, 6):      # 4 is missing: a sequence gap
            await ws.send(json.dumps({"channel": "ticker", "sequence_num": seq, "events": [
                {"type": "update", "tickers": [{"product_id": "BTC-USD",
                                                "best_bid": "82000.5", "best_ask": "82000.6"}]}]}))
            await asyncio.sleep(0.02)
        await ws.wait_closed()           # then silence -> watchdog reconnects

    async with websockets.serve(kraken, "127.0.0.1", 0) as ks, \
            websockets.serve(coinbase, "127.0.0.1", 0) as cs:
        global KRAKEN_URL, COINBASE_URL
        saved = KRAKEN_URL, COINBASE_URL, capture.SILENCE_TIMEOUT, capture.BACKOFF_BASE
        KRAKEN_URL = f"ws://127.0.0.1:{ks.sockets[0].getsockname()[1]}"
        COINBASE_URL = f"ws://127.0.0.1:{cs.sockets[0].getsockname()[1]}"
        capture.SILENCE_TIMEOUT, capture.BACKOFF_BASE = 0.3, 0.05
        try:
            out = await record(seconds=1.5, outdir=tmp)
        finally:
            KRAKEN_URL, COINBASE_URL, capture.SILENCE_TIMEOUT, capture.BACKOFF_BASE = saved
    rows = list(_records(out))           # also proves the file was closed cleanly
    with gzip.open(out, "rt") as fh:
        fh.read()
    # Kraken's fake goes quiet after 30 updates, so the watchdog reconnects and
    # a second connection starts over until the deadline cuts it off.
    k = sum(r.get("v") == "kraken" and "bid" in r for r in rows)
    assert k >= 30, k
    assert sum(r.get("v") == "coinbase" and "bid" in r for r in rows) == 5
    assert any(r.get("v") == "coinbase" and r.get("event") == "disconnect" for r in rows), \
        "silent coinbase connection was not reconnected"
    assert cb_conns["n"] >= 2
    gaps = [r for r in rows if r.get("event") == "seq_gap"]
    assert len(gaps) == 1 and gaps[0]["expected"] == 4 and gaps[0]["got"] == 5, gaps


def demo():
    import tempfile
    assert parse_kraken({"channel": "ticker", "type": "update", "data": [
        {"symbol": "BTC/USD", "bid": Decimal("82000.1"), "ask": Decimal("82000.2")}]},
        "BTC/USD") == [("82000.1", "82000.2")]
    assert parse_kraken({"channel": "heartbeat"}, "BTC/USD") == []
    assert parse_coinbase({"channel": "ticker", "sequence_num": 7, "events": [{"tickers": [
        {"product_id": "BTC-USD", "best_bid": "1", "best_ask": "2"},
        {"product_id": "ETH-USD", "best_bid": "3", "best_ask": "4"}]}]}, "BTC-USD") == ([("1", "2")], 7)

    with tempfile.TemporaryDirectory() as d:
        # A planted 400 ms lead must be found, on the right side, near 400 ms.
        p = os.path.join(d, "leadlag_20261020_090000.jsonl.gz")
        _synthetic(p, lead_ms=400)
        r = analyse(p)
        assert r["leader"] == "kraken" and 250 <= r["peak_lag_ms"] <= 550, r["peak_lag_ms"]
        assert r["peak_corr"] > r["contemporaneous_corr"] + 0.05
        # Negative control: two unrelated walks show no meaningful peak.
        q = os.path.join(d, "leadlag_20261020_100000.jsonl.gz")
        _synthetic(q, lead_ms=400, independent=True, seed=3)
        assert abs(analyse(q)["peak_corr"]) < 0.08
        # Study file: rows accumulate and de-duplicate by file.
        study = os.path.join(d, "leadlag_study.csv")
        append_study(r, study)
        append_study(r, study)
        with open(study) as fh:
            assert len(list(csv.DictReader(fh))) == 1
        # Recorder against fake Kraken + Coinbase servers.
        try:
            import websockets  # noqa: F401
        except ImportError:
            print("skip live recorder test (pip install websockets)")
        else:
            asyncio.run(_fake_servers_test(d))
    print(f"ok — lead-lag: planted 400 ms lead found at {r['peak_lag_ms']} ms, "
          "control shows none, recorder handles both venues, a sequence gap and silence")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--live"]:
        asyncio.run(record(int(args[1]) if len(args) > 1 else 60))
    elif args[:1] == ["--study"]:
        path = args[1] if len(args) > 1 else "data/leadlag_study.csv"
        if not os.path.exists(path):
            sys.exit("no lead-lag study yet: analyse a recording first")
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        print(f"{'start':17}{'minutes':>9}{'leader':>10}{'lag ms':>8}{'peak':>8}{'at zero':>9}")
        for r in rows:
            print(f"{r['start']:17}{r['overlap_min']:>9}{r['leader']:>10}{r['peak_lag_ms']:>8}"
                  f"{r['peak_corr']:>8}{r['contemporaneous_corr']:>9}")
    elif args:
        matches = sorted(glob.glob(args[0]), key=os.path.getmtime)
        if not matches:
            sys.exit(f"no recording matches {args[0]}")
        row = analyse(matches[-1])
        report(row)
        append_study(row, os.path.join(os.path.dirname(matches[-1]) or ".", "leadlag_study.csv"))
    else:
        demo()
