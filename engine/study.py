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
    from engine.research import _ranks, evaluate_orderflow, orderflow_dataset
except ImportError:
    from research import _ranks, evaluate_orderflow, orderflow_dataset

import numpy as np

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


# ---------------------------------------------------------------------------
# Pre-registered test (written 2026-10-30, before any of its data existed)
# ---------------------------------------------------------------------------
# The first week (28 recordings, 2026-10-21 to 10-27) suggested that order
# flow adds more in hours where the mid moves more often (rank correlation
# +0.54, p = 0.004). That came from looking at the data, among several
# comparisons, so the same data can't confirm it. This is the confirmation,
# fixed before the data exists:
#
#   On the FIRST 24 one-hour recordings that start after PREREG_CUTOFF, the
#   rank correlation between moved_pct and adds is positive, with a one-sided
#   permutation p-value below 0.05.
#
# The sample size is fixed in advance: checking the p-value as data arrives
# and stopping when it looks good would make the test meaningless.
PREREG_CUTOFF = "2026-10-30T13:30"       # Toronto time; recordings start after this
PREREG_N = 24                            # three days at eight recordings a day
PREREG_MIN_EVENTS = 50_000               # hour-long recordings have 75k+; skips test runs


def preregistered(rows, n_perm=20_000, seed=0):
    """Score the pre-registered prediction. 'pending' until PREREG_N recordings exist."""
    fresh = [r for r in rows
             if r["start"] > PREREG_CUTOFF and int(r["events"]) >= PREREG_MIN_EVENTS][:PREREG_N]
    out = {"n": len(fresh), "needed": PREREG_N, "verdict": "pending"}
    if len(fresh) < PREREG_N:
        return out
    x = _ranks([float(r["moved_pct"]) for r in fresh])
    y = _ranks([float(r["adds"]) for r in fresh])
    rho = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(seed)
    null = np.array([np.corrcoef(x, rng.permutation(y))[0, 1] for _ in range(n_perm)])
    p = float((null >= rho).mean())                     # one-sided: predicted positive
    out.update(rho=round(rho, 3), p=round(p, 4),
               first=fresh[0]["start"], last=fresh[-1]["start"],
               verdict="confirmed" if rho > 0 and p < 0.05 else "not confirmed")
    return out


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

    t = preregistered(rows)
    print(f"\npre-registered test (2026-10-30): on the first {t['needed']} recordings after "
          f"{PREREG_CUTOFF}, does order flow add more in hours where the mid moves more often?")
    if t["verdict"] == "pending":
        print(f"  pending: {t['n']} of {t['needed']} recordings so far")
    else:
        print(f"  {t['verdict'].upper()}: rank correlation {t['rho']:+.3f}, one-sided p {t['p']:.4f} "
              f"({t['first']} to {t['last']})")


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
    # Pre-registered test: a planted relationship is confirmed, its shuffle
    # isn't, too few recordings stay pending, and recordings from before the
    # cutoff or shorter than an hour are never counted.
    rng = np.random.default_rng(5)
    moved = rng.uniform(8, 24, 30)

    def fake(i, adds_val, start=None, events=120_000):
        return {"start": start or f"2026-11-0{1 + i // 8}T{(i % 8) * 3:02d}:17", "events": events,
                "moved_pct": moved[i], "adds": adds_val}
    planted = [fake(i, 0.004 * moved[i] + rng.normal(0, 0.01)) for i in range(30)]
    early = [fake(i, -1.0, start="2026-10-25T08:00") for i in range(10)]   # before cutoff
    short = [fake(i, -1.0, events=10_000) for i in range(5)]               # test runs
    t = preregistered(early + short + planted)
    assert t["verdict"] == "confirmed" and t["n"] == PREREG_N, t
    shuffled = [dict(r, adds=a) for r, a in zip(planted, rng.permutation([r["adds"] for r in planted]))]
    # Same 24 values, link destroyed: must not confirm (fixed seed, so deterministic).
    sh = preregistered(shuffled)
    assert sh["verdict"] == "not confirmed" and sh["p"] > 0.05, sh
    assert preregistered(planted[:10])["verdict"] == "pending"
    print("ok — study: analyses recordings, caches results, groups by session, "
          "scores the pre-registered test")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        report(update(refresh="--refresh" in sys.argv))
