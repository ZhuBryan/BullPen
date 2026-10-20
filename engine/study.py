"""
study.py: does order flow predict prices, and does it depend on the time of day?

One recording answers "on this hour, what did order flow add?". A study needs
many hours across the day, compared the same way. This runs the order-flow
evaluation (research.py) on every Kraken recording in data/, caches each
result in data/study.csv so a file is only analysed once, and prints:

  * one row per recording, in time order;
  * a summary by session of the day (local time of the recording machine),
    with how many recordings back each number.

The headline column is "adds": the forecast's information coefficient minus
that of the no-model "continue the last move" rule. The model contains that
rule, so "adds" is exactly what order flow tells you beyond the price path.

    python study.py                 # analyse new recordings, print the study
    python study.py --refresh       # re-analyse everything (e.g. after a model change)

Recordings are written by capture.py; scheduled_capture.ps1 makes them on a
timetable (see install_schedule.ps1).
"""

import csv
import glob
import os
import re
import sys
import time
from datetime import datetime

try:
    from engine.research import evaluate_orderflow, orderflow_dataset
except ImportError:
    from research import evaluate_orderflow, orderflow_dataset

FIELDS = ["file", "bytes", "start", "weekday", "hour", "events", "moved_pct",
          "model_auc", "model_ic", "model_ic_t", "continue_ic", "adds", "gross_bp"]

# Sessions of the day, local time. Crypto trades 24/7, but who is awake
# (Asia, Europe, the US) changes how busy the book is, and that's the
# question: does order flow mean more in a quiet book or a busy one?
SESSIONS = [("night 00-06", 0, 6), ("morning 06-12", 6, 12),
            ("afternoon 12-18", 12, 18), ("evening 18-24", 18, 24)]

_STAMP = re.compile(r"_(\d{8})_(\d{6})\.jsonl\.gz$")


def _start(path):
    """Recording start from the filename capture.py gives it (local time)."""
    m = _STAMP.search(path)
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S") if m else None


def analyse(path, horizon=10, window=10):
    """One recording -> one study row (dict of FIELDS), or None if too short."""
    X, move, mid = orderflow_dataset(path, horizon, window)
    res = evaluate_orderflow(X, move, mid, horizon=horizon)
    if not res.get("folds"):
        return None
    m, c = res["model"], res["continue"]
    start = _start(path)
    trades = max(m["pnl"]["trades"], 1)
    return {
        "file": os.path.basename(path), "bytes": os.path.getsize(path),
        "start": start.isoformat(timespec="minutes") if start else "",
        "weekday": start.strftime("%a") if start else "",
        "hour": start.hour if start else "",
        "events": len(move), "moved_pct": round(res["moved_frac"] * 100, 1),
        "model_auc": round(m["auc"], 3), "model_ic": round(m["ic"], 3),
        "model_ic_t": round(m["ic_t"], 1), "continue_ic": round(c["ic"], 3),
        "adds": round(m["ic"] - c["ic"], 3),
        "gross_bp": round(m["pnl"]["gross"] * 1e4 / trades, 4),
    }


def update(data_dir="data", refresh=False, quiet=False):
    """Analyse recordings not yet in data_dir/study.csv; return all rows."""
    cache = os.path.join(data_dir, "study.csv")
    rows = {}
    if os.path.exists(cache) and not refresh:
        with open(cache, newline="") as fh:
            rows = {r["file"]: r for r in csv.DictReader(fh)}
    for path in sorted(glob.glob(os.path.join(data_dir, "kraken_*.jsonl.gz"))):
        name, size = os.path.basename(path), os.path.getsize(path)
        # Re-analyse if the file grew (it was still recording last time).
        if name in rows and int(rows[name]["bytes"]) == size:
            continue
        if not quiet:
            print(f"analysing {name} ({size / 1e6:.1f} MB)...", flush=True)
        t = time.time()
        row = analyse(path)
        if row is None:
            print(f"  skipped {name}: too short to evaluate", file=sys.stderr)
            continue
        rows[name] = row
        if not quiet:
            print(f"  done in {time.time() - t:.0f}s: adds {row['adds']:+.3f} IC")
        # Write after every file, so a long first run that's interrupted
        # keeps what it finished.
        with open(cache, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(sorted(rows.values(), key=lambda r: r["start"]))
    return sorted(rows.values(), key=lambda r: r["start"])


def report(rows):
    if not rows:
        print("no recordings yet: run capture.py (or the scheduled task) first")
        return
    print(f"\n{'start':17}{'day':>4}{'events':>9}{'moved':>7}{'model IC':>10}"
          f"{'continue':>10}{'adds':>8}{'gross bp':>10}")
    for r in rows:
        print(f"{r['start']:17}{r['weekday']:>4}{int(r['events']):9,}{float(r['moved_pct']):6.1f}%"
              f"{float(r['model_ic']):10.3f}{float(r['continue_ic']):10.3f}"
              f"{float(r['adds']):+8.3f}{float(r['gross_bp']):10.4f}")

    print(f"\n{'session':17}{'recordings':>11}{'mean events':>13}{'mean adds':>11}{'min..max adds':>17}")
    for name, lo, hi in SESSIONS:
        grp = [r for r in rows if r["hour"] != "" and lo <= int(r["hour"]) < hi]
        if not grp:
            print(f"{name:17}{0:>11}")
            continue
        adds = [float(r["adds"]) for r in grp]
        ev = sum(int(r["events"]) for r in grp) / len(grp)
        print(f"{name:17}{len(grp):>11}{ev:13,.0f}{sum(adds) / len(adds):+11.3f}"
              f"{min(adds):+9.3f}..{max(adds):+.3f}")
    n = min((sum(1 for r in rows if r["hour"] != "" and lo <= int(r["hour"]) < hi)
             for _, lo, hi in SESSIONS))
    if n < 5:
        print(f"\nfewest recordings in any session: {n}. Treat session differences as "
              "anecdotes until each has about 5; one busy or quiet hour can swing a mean.")
    print("'adds' = model IC minus the no-model 'continue the last move' rule's IC: what "
          "order flow tells you beyond the price path. 'gross bp' is before any fee.")


def demo():
    """Self-check on two synthetic recordings: analyse, cache, re-use the cache."""
    import tempfile
    try:
        from engine.research import synthetic_kraken_capture
    except ImportError:
        from research import synthetic_kraken_capture
    with tempfile.TemporaryDirectory() as d:
        synthetic_kraken_capture(os.path.join(d, "kraken_BTCUSD_20261020_030000.jsonl.gz"),
                                 n=3000, signal=True, seed=1)
        synthetic_kraken_capture(os.path.join(d, "kraken_BTCUSD_20261020_130000.jsonl.gz"),
                                 n=3000, signal=False, seed=2)
        rows = update(d, quiet=True)
        assert [r["hour"] for r in rows] == [3, 13], rows
        sig, noise = rows
        assert float(sig["model_ic"]) > 0.1, "missed the planted signal"
        assert abs(float(noise["model_ic_t"])) < 2.5, "found signal in noise"
        assert os.path.exists(os.path.join(d, "study.csv"))
        cached = update(d, quiet=True)          # nothing new: served from the cache
        assert [r["file"] for r in cached] == [r["file"] for r in rows]
    print("ok — study: analyses recordings, caches results, groups by session")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        report(update(refresh="--refresh" in sys.argv))
