"""
Differential test: the default market simulated in Rust (engine/rust/src/sim.rs)
must match the Python Simulation bit for bit.

The Python simulation is the oracle. For several seeds and run lengths this
compares everything a caller can see: every candle, the whole mid path, the
trade count, and every agent's position and cash (as exact floats, before any
rounding). Then it times both and prints the speedup.

Why exact and not "close": the agents react to the book, so one different
random draw or one float rounded differently changes which orders match, and
every later price diverges. A port that is only approximately right can't be
told apart from one that is wrong.

The conversion from Rust's raw numbers to the API's dict (market_sim.run_default)
is checked first against a Python stand-in, so that part is tested even
without a Rust build.

Skips the Rust part cleanly when the extension is not built:
    cd engine/rust; python -m pip install .
"""

import os
import statistics
import time

try:
    from backend import market_sim
except ImportError:                        # run from engine/
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend import market_sim

CASES = [(7, 1), (7, 50), (1, 300), (42, 2000), (123456789012, 400), (-5, 200)]


def _python_raw(seed, steps):
    """What simulate_default returns, computed by the Python Simulation."""
    sim = market_sim.Simulation(seed=seed)
    sim.run(steps)
    tick = market_sim.TICK
    bars = [(round(b["o"] / tick), round(b["h"] / tick), round(b["l"] / tick),
             round(b["c"] / tick), b["v"]) for b in sim.bars]
    return (bars, sim.last, sim._trades, sim.mid(),
            [(a.position, a.cash) for a in sim.agents], sim.history)


def _agents_exact(seed, steps, engine):
    """(position, cash) per agent, unrounded, for an exact comparison."""
    if engine == "python":
        sim = market_sim.Simulation(seed=seed)
        sim.run(steps)
        return [(a.position, a.cash) for a in sim.agents]
    return market_sim._rust_simulate(seed, steps)[4]


def check_wrapper():
    """run_default's dict building, against a Python stand-in for Rust."""
    real = market_sim._rust_simulate
    market_sim._rust_simulate = _python_raw
    try:
        for seed, steps in CASES[:4]:
            want = market_sim.run_default(seed, steps, engine="python")
            got = market_sim.run_default(seed, steps, engine="rust")
            assert got == want, f"wrapper changes the result, seed {seed}"
    finally:
        market_sim._rust_simulate = real


def demo():
    check_wrapper()
    if market_sim._rust_simulate is None:
        if os.environ.get("BULLPEN_REQUIRE_RUST") == "1":
            raise ImportError("bullpen_engine.simulate_default missing")
        print("ok — run_default wrapper matches Python; skip Rust simulator "
              "(not built: cd engine/rust; python -m pip install .)")
        return

    for seed, steps in CASES:
        py_out, py_hist = market_sim.run_default(seed, steps, engine="python")
        rs_out, rs_hist = market_sim.run_default(seed, steps, engine="rust")
        assert rs_hist == py_hist, f"mid path differs, seed {seed} steps {steps}"
        assert rs_out == py_out, f"result differs, seed {seed} steps {steps}"
        assert _agents_exact(seed, steps, "rust") == _agents_exact(seed, steps, "python"), \
            f"unrounded cash differs, seed {seed} steps {steps}"

    def timed(engine, seed=7, steps=2000, repeats=5):
        market_sim.run_default(seed, 200, engine=engine)            # warmup
        times = []
        for _ in range(repeats):
            t = time.perf_counter()
            market_sim.run_default(seed, steps, engine=engine)
            times.append(time.perf_counter() - t)
        return statistics.median(times)

    py_t, rs_t = timed("python"), timed("rust")
    print(f"ok — Rust simulator identical to Python on {len(CASES)} runs "
          f"(every candle, mid and cash bit); 2000 steps: Python {py_t * 1e3:.1f} ms, "
          f"Rust {rs_t * 1e3:.2f} ms, {py_t / rs_t:.0f}x faster")


if __name__ == "__main__":
    demo()
