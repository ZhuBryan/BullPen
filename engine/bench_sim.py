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

    python -m engine.bench_sim
"""

import os
import statistics
import subprocess
import sys

STEPS, REPEATS = 2000, 5

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


if __name__ == "__main__":
    main()
