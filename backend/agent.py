"""
agent.py — an autonomous strategy search that discounts its own findings.

The loop is ordinary: propose a strategy, backtest it, look at the result,
propose a better one. Any parameter sweep does that. What makes this defensible
is the critic, and the specific thing the critic knows:

    IF YOU TRY ENOUGH STRATEGIES, ONE WILL LOOK BRILLIANT BY LUCK.

Search 200 variants of a coin flip and the best will show a Sharpe near 2.5.
Report that number and you have reported nothing. This is the single most
common way quant research produces garbage, and an agent is *especially* prone
to it because searching is exactly what an agent is good at. An agent without
this correction is a machine for manufacturing false positives at scale.

So every result is passed through the Deflated Sharpe Ratio (Bailey and Lopez
de Prado). Two ingredients:

  1. EXPECTED MAXIMUM UNDER THE NULL. Given N independent trials on strategies
     with no real edge, the best observed Sharpe is still positive, and grows
     with N. That benchmark is what a candidate must beat, not zero.

  2. THE NON-NORMALITY HAIRCUT. Sharpe assumes normal returns. Real strategy
     returns are skewed and fat-tailed, which makes the estimate less certain
     than it looks, so the test statistic is widened accordingly.

DSR is then the probability the true Sharpe exceeds zero, *after* accounting
for how hard you looked. A DSR of 0.95 means it survives scrutiny. A Sharpe of
2.0 found on trial 300 will not.

Evaluation is out-of-sample: parameters are chosen on the training window only,
and scored on a held-out later window the search never touched.

Run `python backend/agent.py` for the self-check.
"""

import math
import random
from statistics import NormalDist

try:
    from backend.backtest import run_backtest
    from backend.strategies import STRATEGIES
except ImportError:                        # running this file directly
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend.backtest import run_backtest
    from backend.strategies import STRATEGIES

EULER = 0.5772156649015329
_N = NormalDist()


# ---------------------------------------------------------------------------
# The critic
# ---------------------------------------------------------------------------
def expected_max_sharpe(n_trials: int, n_obs: int) -> float:
    """Best Sharpe you would expect from `n_trials` strategies with NO edge.

    This is the bar a candidate has to clear. It rises with the number of
    trials, which is the entire point: searching harder raises the bar.
    """
    n_trials = max(int(n_trials), 1)
    if n_trials == 1 or n_obs < 2:
        return 0.0
    sigma = 1.0 / math.sqrt(max(n_obs - 1, 1))     # std error of a Sharpe estimate
    a = _N.inv_cdf(1 - 1.0 / n_trials)
    b = _N.inv_cdf(1 - 1.0 / (n_trials * math.e))
    return sigma * ((1 - EULER) * a + EULER * b)


def deflated_sharpe(sharpe: float, n_trials: int, n_obs: int,
                    skew: float = 0.0, kurtosis: float = 3.0,
                    n_obs_trials: int | None = None) -> float:
    """Probability the true Sharpe beats the selection-adjusted benchmark SR0.

    Note it is P(true SR > SR0), NOT P(true SR > 0): the bar already includes
    what luck would produce over `n_trials` attempts.

    n_obs        length of the sample being scored.
    n_obs_trials length of the sample the SEARCH ran on. The trial Sharpes whose
                 maximum SR0 describes were estimated over that window, so the
                 standard error in SR0 must use it. Defaults to n_obs, which is
                 only correct when you searched and scored on the same data.

    Returns a value in [0, 1]. Above ~0.95 is the usual bar for "believable".
    """
    if n_obs < 3:
        return 0.0
    sr0 = expected_max_sharpe(n_trials, n_obs_trials or n_obs)
    # Widen the uncertainty for skew and fat tails.
    denom = 1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe ** 2
    if denom <= 0:
        return 0.0
    z = (sharpe - sr0) * math.sqrt(n_obs - 1) / math.sqrt(denom)
    return float(_N.cdf(z))


def moments(returns):
    """(sharpe_per_observation, skew, kurtosis) computed without scipy."""
    n = len(returns)
    if n < 3:
        return 0.0, 0.0, 3.0
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / n
    sd = math.sqrt(var)
    if sd == 0:
        return 0.0, 0.0, 3.0
    skew = sum(((r - mean) / sd) ** 3 for r in returns) / n
    kurt = sum(((r - mean) / sd) ** 4 for r in returns) / n
    return mean / sd, skew, kurt


def critique(returns, n_trials, threshold=0.95, n_obs_trials=None):
    """The verdict, in numbers and in plain English.

    n_obs_trials: length of the window the search ran on (see deflated_sharpe).
    """
    n = len(returns)
    sr_obs, skew, kurt = moments(returns)
    sr_annual = sr_obs * math.sqrt(252)
    dsr = deflated_sharpe(sr_obs, n_trials, n, skew, kurt, n_obs_trials)
    bar = expected_max_sharpe(n_trials, n_obs_trials or n) * math.sqrt(252)
    passed = dsr >= threshold and sr_annual > bar

    if passed:
        reason = (f"Sharpe {sr_annual:.2f} clears the {bar:.2f} expected from "
                  f"{n_trials} trials of pure luck (DSR {dsr:.3f}).")
    elif sr_annual <= bar:
        reason = (f"Sharpe {sr_annual:.2f} does not even beat the {bar:.2f} you would "
                  f"expect from {n_trials} lucky trials. This is a search artifact.")
    else:
        reason = (f"Sharpe {sr_annual:.2f} beats the {bar:.2f} luck benchmark but "
                  f"DSR is only {dsr:.3f}, short of {threshold}. Not enough evidence.")

    return {"sharpe": round(sr_annual, 3), "deflated_sharpe": round(dsr, 4),
            "luck_benchmark": round(bar, 3), "n_trials": n_trials,
            "n_obs": n, "skew": round(skew, 3), "kurtosis": round(kurt, 3),
            "passed": bool(passed), "reason": reason}


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------
def _returns_from_equity(equity):
    return [equity[i] / equity[i - 1] - 1 for i in range(1, len(equity))
            if equity[i - 1]]


class ResearchAgent:
    """Searches strategy parameters, then refuses to trust itself."""

    def __init__(self, prices, budget=40, seed=0, train_frac=0.7,
                 realistic_fills=True, explore=0.35):
        self.prices = prices
        self.budget, self.explore = budget, explore
        self.rnd = random.Random(seed)
        self.realistic_fills = realistic_fills
        cut = int(len(prices) * train_frac)
        self.train, self.test = prices.iloc[:cut], prices.iloc[cut:]
        self.trials = 0
        self.history = []
        self.best = None            # best by TRAIN score; test stays untouched
        self.candidates = [s for s in STRATEGIES if STRATEGIES[s]["params"]]

    # ---- tools the agent can call -------------------------------------
    def _sample(self, p):
        """Draw one parameter, respecting whether it is integer or continuous.

        Sampling Bollinger's num_std (0.5 to 4.0) with randint would floor the
        range to 0..4 and can pick 0, which collapses the bands onto the moving
        average and makes the strategy meaningless while still burning a trial.
        """
        lo, hi = p["min"], p["max"]
        if isinstance(lo, float) or isinstance(hi, float) or isinstance(p["default"], float):
            return round(self.rnd.uniform(float(lo), float(hi)), 3)
        return self.rnd.randint(int(lo), int(hi))

    def propose(self):
        """Explore a random strategy, or exploit by nudging the best so far."""
        if self.best is None or self.rnd.random() < self.explore:
            sid = self.rnd.choice(self.candidates)
            return sid, {p["name"]: self._sample(p) for p in STRATEGIES[sid]["params"]}
        sid, params = self.best["strategy"], dict(self.best["params"])
        spec = {p["name"]: p for p in STRATEGIES[sid]["params"]}
        for k in params:                      # jitter around the incumbent
            p = spec[k]
            lo, hi = float(p["min"]), float(p["max"])
            step = (hi - lo) / 10.0
            nudged = float(params[k]) + self.rnd.uniform(-step, step)
            nudged = max(lo, min(hi, nudged))
            params[k] = (round(nudged, 3)
                         if isinstance(p["default"], float) else int(round(nudged)))
        return sid, params

    def backtest(self, sid, params, prices):
        try:
            positions = STRATEGIES[sid]["func"](prices, **params)
            return run_backtest(prices, positions, realistic_fills=self.realistic_fills)
        except Exception:
            return None

    # ---- the loop ------------------------------------------------------
    def run(self):
        for _ in range(self.budget):
            sid, params = self.propose()
            res = self.backtest(sid, params, self.train)
            self.trials += 1                  # every look counts, even a bad one
            if res is None:
                continue
            score = res["metrics"]["sharpe"]
            row = {"strategy": sid, "params": params, "train_sharpe": score}
            self.history.append(row)
            if self.best is None or score > self.best["train_sharpe"]:
                self.best = row
        return self.report()

    def report(self):
        if not self.best:
            return {"found": False, "reason": "no valid strategy was produced",
                    "trials": self.trials}

        # The held-out window, touched exactly once, at the end.
        oos = self.backtest(self.best["strategy"], self.best["params"], self.test)
        if oos is None:
            return {"found": False, "reason": "best strategy failed out of sample",
                    "trials": self.trials}

        # The search ran on the training window, so the luck benchmark is sized
        # on that length, not on the shorter held-out window being scored.
        verdict = critique(_returns_from_equity(oos["equity"]), self.trials,
                           n_obs_trials=len(self.train))
        return {
            "found": True,
            "strategy": self.best["strategy"],
            "params": self.best["params"],
            "trials": self.trials,
            "train_sharpe": round(self.best["train_sharpe"], 3),
            "out_of_sample": {
                "return_pct": oos["metrics"]["total_return_pct"],
                "sharpe": oos["metrics"]["sharpe"],
                "max_drawdown_pct": oos["metrics"]["max_drawdown_pct"],
                "num_trades": oos["metrics"]["num_trades"],
            },
            "critic": verdict,
        }


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
def _draw(rnd, n, mu, sd):
    """Independent return series, so a test does not depend on draw order."""
    return [rnd.gauss(mu, sd) for _ in range(n)]


def demo():
    # ---- the core property: more searching raises the bar ----
    bars = [expected_max_sharpe(n, 500) for n in (1, 10, 100, 1000)]
    assert bars[0] == 0.0, "a single trial needs no correction"
    assert bars[1] < bars[2] < bars[3], f"bar must rise with trials: {bars}"

    # ---- and lowers belief in the same result ----
    dsrs = [deflated_sharpe(0.1, n, 500) for n in (1, 10, 100, 10_000)]
    assert dsrs[0] > dsrs[1] > dsrs[2] > dsrs[3], f"DSR must fall with trials: {dsrs}"
    assert 0.0 <= min(dsrs) and max(dsrs) <= 1.0, "DSR must be a probability"

    # a strong, single-trial result is believable
    assert deflated_sharpe(0.15, 1, 1000) > 0.95
    # the same number after a big search is not
    assert deflated_sharpe(0.15, 5000, 1000) < 0.95

    # fat tails must reduce confidence
    thin = deflated_sharpe(0.1, 10, 500, skew=0.0, kurtosis=3.0)
    fat = deflated_sharpe(0.1, 10, 500, skew=-1.0, kurtosis=12.0)
    assert fat < thin, "fat tails and negative skew should lower DSR"

    # ---- the critic rejects noise, which is the whole job ----
    rnd = random.Random(1)
    noise = [rnd.gauss(0, 0.01) for _ in range(600)]
    v = critique(noise, n_trials=200)
    assert not v["passed"], f"noise passed the critic: {v}"
    assert "artifact" in v["reason"] or "evidence" in v["reason"]

    # a genuinely strong edge should pass even after a big search
    strong = _draw(random.Random(2), 600, 0.0015, 0.005)
    assert critique(strong, n_trials=1)["passed"], "a real edge was rejected"
    assert critique(strong, n_trials=5_000)["passed"], \
        "a genuinely large edge should survive selection correction"

    # the point of the whole file: a MARGINAL edge is believable once, and is
    # not believable once you admit you searched 5000 times for it.
    marginal = _draw(random.Random(4), 600, 0.0011, 0.01)
    once = critique(marginal, n_trials=1)
    searched = critique(marginal, n_trials=5_000)
    assert once["passed"], f"marginal edge should pass a single honest test: {once}"
    assert not searched["passed"], "selection bias was not applied"
    assert searched["deflated_sharpe"] < once["deflated_sharpe"]
    assert searched["luck_benchmark"] > once["luck_benchmark"]
    assert searched["sharpe"] == once["sharpe"], "the data did not change, only the honesty"

    # degenerate inputs must not explode
    assert critique([], 10)["deflated_sharpe"] == 0.0
    assert critique([0.0] * 50, 10)["sharpe"] == 0.0

    # ---- end to end on real (or sample) prices ----
    from backend.data import get_prices
    prices, _ = get_prices("AAPL", "2020-01-01", "2024-01-01")
    agent = ResearchAgent(prices, budget=25, seed=3)
    out = agent.run()
    assert out["found"], out
    assert out["trials"] == 25, "every proposal must count as a trial"
    assert out["critic"]["n_trials"] == 25
    assert set(out["params"]) == {p["name"] for p in STRATEGIES[out["strategy"]]["params"]}
    # the agent must not have peeked: test window is separate from train
    assert len(agent.train) + len(agent.test) == len(prices)
    assert agent.train.index[-1] < agent.test.index[0], "train/test overlap in time"

    print(f"ok — critic rejects noise, discounts search "
          f"(bar {bars[1]:.4f} -> {bars[3]:.4f}); agent ran {out['trials']} trials, "
          f"verdict passed={out['critic']['passed']}")


if __name__ == "__main__":
    demo()
