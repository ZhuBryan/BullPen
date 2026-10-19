"""
bench_sim.py — how much faster does the APP get on the Rust engine?

bench.py measures the matching engine alone (12.7x). This measures what a
user actually waits for: the agent-based market simulation, which spends
much of its time in Python agent logic that Rust doesn't touch. By Amdahl's
law the end-to-end gain is bounded by the share of time spent matching, so
expect far less than 12.7x, and quote whatever this prints, not the engine
number.

Each engine runs in a fresh process (fastbook picks the engine at import),
on the same seed, so both do identical work. Run from the repo root:

    python -m engine.bench_sim                     # Python vs Rust, end to end
    python -m engine.bench_sim --profile           # where the time goes (fastest engine)
    python -m engine.bench_sim --profile --engine python

--profile answers "what share of the run is the book?", two ways:

  * cProfile, every function sorted into a category: matching (the book,
    including the builtins it calls, and the Rust methods themselves),
    agent logic, simulation loop, builtins called from Python code (sum,
    sorted, list.append...), and random. cProfile charges a fixed cost to
    every Python call, so it OVERSTATES the share of code made of many tiny
    Python calls. Read it for the ranking, not the exact split.
  * Replay, which has no profiler in it: every call the simulation made on
    the book (limit, cancel, best_bid, depth, events_since...) is recorded,
    then replayed on a fresh book in a bare loop and timed. Replay time over
    simulation time is the book's real share s, and Amdahl's law gives the
    best end-to-end gain an engine k times faster could buy:
        speedup = 1 / ((1 - s) + s / k)
    It also counts the calls, because every Python->Rust call pays a fixed
    conversion cost no matter how little work it does. As a cross-check the
    simulation is also run against a "tape" that answers each book call with
    the recorded result and does no work: that times the Python side alone,
    and the two parts should roughly add up to the whole.
"""

import argparse
import os
import statistics
import subprocess
import sys

STEPS, REPEATS = 2000, 5
ENGINE_SPEEDUP = 12.7     # bench.py, matching alone, Rust vs Python

CHILD = f"""
import time
from engine.fastbook import ENGINE
from backend.market_sim import Simulation
Simulation(seed=7).run(200)                       # warmup
times = []
for _ in range({REPEATS}):
    t = time.perf_counter()
    out = Simulation(seed=7).run({STEPS})
    times.append(time.perf_counter() - t)
print(ENGINE, out["trades"], *times)
"""


def run(engine):
    env = {**os.environ, "BULLPEN_ENGINE": engine}
    line = subprocess.run([sys.executable, "-c", CHILD], env=env, check=True,
                          capture_output=True, text=True).stdout.split()
    return line[0], int(line[1]), [float(x) for x in line[2:]]


def main():
    results = {}
    for want in ("python", "rust"):
        got, trades, times = run(want)
        if got != want:
            print(f"{want}: not available (got {got}); build it with "
                  f"`python -m pip install .` in engine/rust")
            continue
        results[got] = (trades, statistics.median(times))
        print(f"{got:6}  {trades} trades  median {statistics.median(times):.3f}s  "
              f"(min {min(times):.3f}, max {max(times):.3f}, {REPEATS} runs of {STEPS} steps)")
    if len(results) == 2:
        (tp, py), (tr, rs) = results["python"], results["rust"]
        # Same seed, same agents: if the engines differ in behaviour the trade
        # counts differ, and the timing comparison would be meaningless.
        assert tp == tr, f"engines did different work: {tp} vs {tr} trades"
        print(f"\nend-to-end speedup: {py / rs:.2f}x (identical work: {tp} trades each)")


# ---------------------------------------------------------------------------
# --profile (runs in a child process so BULLPEN_ENGINE picks the engine)
# ---------------------------------------------------------------------------
CATEGORIES = ("matching", "agent logic", "sim loop", "builtins", "random", "other")
_AGENT_FUNCS = {"act", "fair_value", "sigma", "quotes", "on_fill", "equity",
                "pnl_parts", "__init__"}


def _category(func):
    """Sort one cProfile entry (filename, line, name) into a category."""
    filename, _, name = func
    path = filename.replace("\\", "/")
    if "bullpen_engine" in name or path.endswith("engine/orderbook.py"):
        return "matching"
    if path.endswith("random.py") or "_random" in name:
        return "random"
    if path.endswith("backend/market_sim.py"):
        # Agent methods vs the Simulation driver. Both live in one file, so
        # split on the method name (Simulation has no method with these names).
        return "agent logic" if name in _AGENT_FUNCS else "sim loop"
    if filename == "~":
        return "builtins"
    return "other"


class _Recorder:
    """Stands in front of a Book and logs every call and its result."""

    def __init__(self, book, log):
        self._book, self._log = book, log

    def __getattr__(self, name):
        attr = getattr(self._book, name)
        if not callable(attr):
            return attr

        def call(*args):
            result = attr(*args)
            self._log.append((name, args, result))
            return result
        return call


class _Tape:
    """A fake book that answers every call with the recorded result and does
    no work at all: an infinitely fast book with free calls. A simulation run
    against it times the Python side alone (minus the tape's own small cost,
    measured separately and subtracted)."""

    def __init__(self, log):
        nxt = iter([r for _, _, r in log]).__next__
        for name in {n for n, _, _ in log}:
            setattr(self, name, lambda *a, _next=nxt: _next())


def profile_child(steps, repeats):
    import cProfile
    import gc
    import pstats
    import time

    from backend import market_sim
    from engine.fastbook import ENGINE, Book

    real_book = market_sim.Book
    market_sim.Simulation(seed=7).run(200)                     # warmup

    # Record every book call the simulation makes, with its result.
    log = []
    market_sim.Book = lambda: _Recorder(real_book(), log)
    try:
        rec_out = market_sim.Simulation(seed=7).run(steps)
    finally:
        market_sim.Book = real_book
    # The recording holds every result the book returned, event tuples
    # included. Freeze it out of the garbage collector's view, or every
    # collection during the timed runs below would walk it too, slowing them
    # by an amount a normal run never pays.
    gc.collect()
    gc.freeze()

    def sim():
        market_sim.Book = real_book
        return market_sim.Simulation(seed=7).run(steps)

    def python_side():
        market_sim.Book = lambda: _Tape(log)
        try:
            return market_sim.Simulation(seed=7).run(steps)
        finally:
            market_sim.Book = real_book

    def tape_cost():                    # the tape's own overhead, same calls
        tape = _Tape(log)
        for name, args, _ in log:
            getattr(tape, name)(*args)

    def book_only():                    # the same calls on a real book, bare loop
        fresh = Book()
        calls = [(getattr(fresh, name), args) for name, args, _ in log]
        t = time.perf_counter()
        for f, args in calls:
            f(*args)
        return time.perf_counter() - t

    # Interleaved, so all four see the same machine state (laptops and VMs
    # drift by 10-20% between runs; a ratio of medians taken minutes apart
    # would mostly measure that).
    t_sim, t_py, t_tape, t_book = [], [], [], []
    for _ in range(repeats):
        t = time.perf_counter(); out = sim(); t_sim.append(time.perf_counter() - t)
        t = time.perf_counter(); py_out = python_side(); t_py.append(time.perf_counter() - t)
        t = time.perf_counter(); tape_cost(); t_tape.append(time.perf_counter() - t)
        t_book.append(book_only())
    assert rec_out == out == py_out, "recording or tape changed the run"
    med = statistics.median
    sim_t, book_t = med(t_sim), med(t_book)
    py_t = max(med(t_py) - med(t_tape), 0.0)

    # cProfile, one run.
    prof = cProfile.Profile()
    prof.enable()
    sim()
    prof.disable()
    st = pstats.Stats(prof)
    total = sum(v[2] for v in st.stats.values())
    cat = dict.fromkeys(CATEGORIES, 0.0)
    for func, (cc, nc, tt, ct, callers) in st.stats.items():
        c = _category(func)
        if c in ("builtins", "other") and callers:
            # A builtin's time belongs to whoever called it: bisect or
            # deque.popleft called by the book is matching, sum() called by
            # an agent is "builtins". pstats keeps tottime per caller edge.
            # (Same for generated code such as a dataclass __init__.)
            for caller, edge in callers.items():
                cc_ = _category(caller)
                cat[cc_ if cc_ in ("matching", "random") else c] += edge[2]
        else:
            cat[c] += tt
    counts = {}
    for name, _, _ in log:
        counts[name] = counts.get(name, 0) + 1

    print(f"engine: {ENGINE}   {steps} steps, seed 7, {out['trades']} trades")
    print(f"simulation, no profiler: median {sim_t * 1000:.1f} ms "
          f"(min {min(t_sim) * 1000:.1f}, max {max(t_sim) * 1000:.1f}, {repeats} runs)\n")
    print(f"cProfile breakdown (tottime, {total * 1000:.0f} ms under the profiler; "
          f"it inflates many-small-call code, so trust the ranking more than the split):")
    for c in CATEGORIES:
        print(f"  {c:12} {cat[c] * 1000:8.1f} ms  {100 * cat[c] / total:5.1f}%")
    print("\ntop 15 functions by tottime:")
    print(f"  {'tottime':>9} {'calls':>8}  function")
    top = sorted(st.stats.items(), key=lambda kv: -kv[1][2])[:15]
    for (fn, line, name), (cc, nc, tt, ct, _) in top:
        where = f"{os.path.basename(fn)}:{line} {name}" if fn != "~" else name
        print(f"  {tt * 1000:7.1f}ms {nc:8d}  {where}")

    s = min(book_t / sim_t, 1.0)
    print("\nwithout a profiler (medians, interleaved):")
    print(f"  the book calls alone, replayed:                {book_t * 1000:6.1f} ms "
          f"= {100 * book_t / sim_t:.1f}%   <- book share s")
    print(f"  Python side alone (book answered from a tape): {py_t * 1000:6.1f} ms "
          f"= {100 * py_t / sim_t:.1f}%   (cross-check)")
    print(f"  (the two should roughly add up to the simulation, {sim_t * 1000:.1f} ms; "
          f"on a noisy machine rerun with --repeats 25)")
    print(f"book calls per run (each is one Python->{ENGINE} crossing): {len(log)}")
    print("  " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))
    if ENGINE == "python":
        best = 1 / ((1 - s) + s / ENGINE_SPEEDUP)
        print(f"\nAmdahl with book share s = {100 * s:.1f}%: a book {ENGINE_SPEEDUP}x faster "
              f"gives at most {best:.2f}x end to end; an infinitely fast one "
              f"{1 / max(1 - s, 1e-9):.2f}x. Per-call FFI cost is not in that; "
              f"measure with `python -m engine.bench_sim`.")
    else:
        print(f"\nPython code is {100 * (1 - s):.1f}% of the run: an even faster "
              f"book could gain at most {1 / max(1 - s, 1e-9):.2f}x more.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--profile", action="store_true",
                    help="print where the time goes instead of comparing engines")
    ap.add_argument("--engine", choices=("rust", "python"),
                    help="engine for --profile (default: rust if built, else python)")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child:
        profile_child(a.steps, a.repeats)
    elif a.profile:
        env = dict(os.environ)
        if a.engine:
            env["BULLPEN_ENGINE"] = a.engine
        subprocess.run([sys.executable, "-m", "engine.bench_sim", "--child",
                        "--steps", str(a.steps), "--repeats", str(a.repeats)],
                       env=env, check=True)
    else:
        main()
