"""
test_bullpen.py — plain-python tests for the Bullpen arena backend.

No pytest needed. Run from the project root:

    python backend/test_bullpen.py

Covers:
  1. DSL parsing — valid rules parse, broken rules fail with readable errors
  2. rule_positions — stateful enter/exit logic on a tiny handmade DataFrame
  3. Arena — state shape, lazy replay clock advancing, season auto-restart
  4. Agents — add (valid + invalid), persistence, delete (custom vs default)
  5. Custom backtest through the real API handler
  9. Optimizer — grid search returns correct combos + best
  10. Walk-forward — folds, combined out-of-sample curve, error cases
  11. Monte Carlo — percentile ordering + risk stats
  12. Analysis API endpoints (/api/optimize, /api/walkforward, /api/montecarlo)
"""

import os
import sys
import tempfile
import time

# Allow running this file directly: put the project root on sys.path so
# `import backend.xxx` works the same as when uvicorn runs the app.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from backend import rules
from backend.analysis import optimize_strategy, walk_forward, monte_carlo
from backend.arena import Arena, DEFAULT_AGENTS, START_OFFSET
from backend.backtest import run_backtest
from backend.data import _generate_sample_data
from backend.strategies import STRATEGIES, bollinger_bands, sma_crossover

PASSED = 0


def check(condition, label):
    global PASSED
    assert condition, f"FAILED: {label}"
    PASSED += 1
    print(f"  ok - {label}")


# ===========================================================================
print("\n[1] DSL parsing — valid rules")
# ===========================================================================
VALID_RULES = [
    "CLOSE > 0",
    "close < sma(200)",                              # case-insensitive
    "RSI(14) < 30 AND CLOSE > SMA(200)",
    "SMA(20) > SMA(50) OR MOM(60) > 0",
    "(RSI(14) < 30) AND (CLOSE > SMA(200) OR MOM(20) > 0.05)",
    "EMA(12) >= EMA(26)",
    "1 > 0",                                          # constant condition
    "MOM(60) > 0 AND MOM(20) > 0 OR RSI(2) < 10",     # AND binds tighter
]
for text in VALID_RULES:
    ok, err = rules.validate_rules(text, "CLOSE < 0")
    check(ok, f"valid: {text!r}")

# ===========================================================================
print("\n[2] DSL parsing — invalid rules give human-readable errors")
# ===========================================================================
INVALID_RULES = [
    "",                          # empty
    "CLOSE >",                   # missing right side
    "SMA(20) SMA(50)",           # missing operator
    "BANANA(10) > 5",            # unknown indicator
    "SMA(0) > 1",                # period below 1
    "SMA(501) > 1",              # period above 500
    "SMA(20.5) > 1",             # non-integer period
    "RSI(14) < 30 AND",          # dangling AND
    "(CLOSE > SMA(50)",          # unbalanced paren
    "CLOSE = 100",               # '=' isn't a valid operator
    "CLOSE > 100 200",           # junk after a complete rule
    "SMA > 50",                  # SMA without (n)
    "(RSI(14) < 30) > 5",        # comparing a condition to a number
    "AND CLOSE > 0",             # starts with a keyword
]
for text in INVALID_RULES:
    ok, err = rules.validate_rules(text, "CLOSE < 0")
    check(not ok, f"rejected: {text!r}")
    check(isinstance(err, str) and len(err) > 10, f"  ...with a real message: {err[:60]!r}")

# Bad sell rule is also caught (and labelled as the sell rule)
ok, err = rules.validate_rules("CLOSE > 0", "RSI() > 70")
check(not ok and err.lower().startswith("sell rule"), f"bad sell rule flagged: {err[:60]!r}")

# ===========================================================================
print("\n[3] rule_positions — stateful enter/exit on a handmade DataFrame")
# ===========================================================================
# 8 days of prices chosen so we can verify every transition by hand.
closes = [100, 95, 89, 92, 105, 112, 88, 96]
idx = pd.bdate_range("2024-01-01", periods=len(closes))
df = pd.DataFrame({"Close": closes}, index=idx)

# Buy when price drops below 90, sell when it rises above 110. Walking it:
# 100 flat, 95 flat, 89 BUY->long, 92 long, 105 long, 112 SELL->flat,
# 88 BUY->long, 96 long.
pos = rules.rule_positions(df, "CLOSE < 90", "CLOSE > 110")
check(list(pos) == [0, 0, 1, 1, 1, 0, 1, 1], "enter/exit state machine matches hand-walk")
check(list(pos.index) == list(df.index), "positions share the prices' index")

# Buy & hold style: always-true buy, never-true sell -> long from day 0.
pos = rules.rule_positions(df, "CLOSE > 0", "CLOSE < 0")
check(list(pos) == [1] * len(df), "always-true buy rule = long every day")

# Warm-up: SMA(3) is NaN for the first 2 days, so the comparison is false
# there. SMA(3) values: -, -, 94.67, 92, 95.33, 103, 101.67, 98.67.
pos = rules.rule_positions(df, "CLOSE > SMA(3)", "CLOSE < SMA(3)")
check(list(pos[:2]) == [0, 0], "NaN warm-up days can't trigger a buy")
# Day 4: 105 > 95.33 -> buy. Day 6: 88 < 101.67 -> sell.
check(pos.iloc[4] == 1 and pos.iloc[6] == 0, "SMA cross enters day 4, exits day 6")

# AND/OR precedence: 'A OR B AND C' must parse as 'A OR (B AND C)'.
# A = CLOSE > 99 (days 0,4,5), B = CLOSE < 90 (days 2,6), C = CLOSE > 200 (never).
# So the whole thing is just A. If precedence were ((A OR B) AND C) it'd never fire.
sig = rules.rule_signal(df, "CLOSE > 99 OR CLOSE < 90 AND CLOSE > 200")
check(list(sig) == [c > 99 for c in closes], "AND binds tighter than OR")

# ===========================================================================
print("\n[4] Arena — state shape, lazy clock, agents")
# ===========================================================================
# Inject synthetic data + a temp agents file so tests are offline and don't
# touch the real data/agents.json. Speed the clock up 20x for the test.
sample = _generate_sample_data("ARENATEST", "2023-01-01", "2024-06-01")


def fake_loader(ticker):
    return sample.copy(), "sample"


tmp_agents = os.path.join(tempfile.mkdtemp(), "agents.json")
arena = Arena(ticker="TEST", seconds_per_day=0.1, agents_file=tmp_agents,
              data_loader=fake_loader)

state = arena.get_state()
check(state["ticker"] == "TEST", "state.ticker")
check(state["data_source"] == "sample", "state.data_source")
check(state["season"] == 1, "first season is 1")
check(state["total_days"] == len(sample), "state.total_days")
check(state["day_index"] >= START_OFFSET, "race starts at the warm-up offset")
check(len(state["dates"]) == state["day_index"] + 1, "dates revealed up to day_index only")
check(state["date"] == state["dates"][-1], "state.date is the current day")
check(len(state["agents"]) == 5, "5 default personalities")

agent = state["agents"][0]
for key in ("id", "name", "emoji", "tagline", "is_default", "buy_rule",
            "sell_rule", "equity_series", "equity", "return_pct", "position", "rank"):
    check(key in agent, f"agent has '{key}'")
check(agent["rank"] == 1, "agents sorted by rank, best first")
check([a["rank"] for a in state["agents"]] == list(range(1, 6)), "ranks are 1..N")
check(all(a["position"] in ("LONG", "CASH") for a in state["agents"]), "position is LONG/CASH")
check(all(len(a["equity_series"]) == len(state["dates"]) for a in state["agents"]),
      "equity series aligned with revealed dates")
# Warm-up fairness: everyone is exactly $10,000 at the starting line.
check(all(a["equity_series"][START_OFFSET] == 10000.0 for a in state["agents"]),
      "all agents start the race at $10,000")

# Lazy clock: wait a bit, day_index must advance (no background threads).
di_before = arena.get_state()["day_index"]
time.sleep(0.35)
state = arena.get_state()
di_after = state["day_index"]
check(di_after > di_before, f"replay clock advanced ({di_before} -> {di_after})")

# By now the race is a few days in, so HODL (always-true buy rule) has had
# time to enter on day start+1 — and it never exits.
hodl = next(a for a in state["agents"] if a["id"] == "hodl")
check(hodl["position"] == "LONG", "HODL never sells")

# ===========================================================================
print("\n[5] Arena — add / persist / delete agents")
# ===========================================================================
new_agent = arena.add_agent("Crossy", "🦀", "SMA(10) > SMA(30)", "SMA(10) < SMA(30)")
check(new_agent["name"] == "Crossy" and not new_agent["is_default"], "custom agent created")
check(new_agent["equity"] == 10000.0, "joins with exactly $10,000")
check(all(v == 10000.0 for v in new_agent["equity_series"]),
      "flat $10,000 series before/at its join day")
check("rank" in new_agent, "returned agent includes its rank")
check(len(arena.get_state()["agents"]) == 6, "arena now has 6 agents")

# Persisted to disk?
import json
with open(tmp_agents, encoding="utf-8") as f:
    saved = json.load(f)
check(len(saved) == 1 and saved[0]["name"] == "Crossy", "custom agent saved to agents.json")
check({"id", "name", "emoji", "buy_rule", "sell_rule", "created_at"} <= set(saved[0]),
      "agents.json has id/name/emoji/rules/created_at")

# A fresh Arena (same file) should resurrect the saved agent.
arena2 = Arena(ticker="TEST", seconds_per_day=0.1, agents_file=tmp_agents,
               data_loader=fake_loader)
check(any(a["id"] == saved[0]["id"] for a in arena2.get_state()["agents"]),
      "saved agent survives a restart")

# Invalid creations are rejected with ValueError.
for bad_args, label in [
    (("", "🦀", "CLOSE > 0", "CLOSE < 0"), "empty name rejected"),
    (("x" * 25, "🦀", "CLOSE > 0", "CLOSE < 0"), "25-char name rejected"),
    (("Bad", "🦀", "BANANA > 0", "CLOSE < 0"), "bad buy rule rejected"),
    (("Bad", "🦀", "CLOSE > 0", "CLOSE <"), "bad sell rule rejected"),
]:
    try:
        arena.add_agent(*bad_args)
        check(False, label)
    except ValueError:
        check(True, label)

# Delete: custom OK, default refused, unknown -> KeyError.
arena.delete_agent(new_agent["id"])
check(len(arena.get_state()["agents"]) == 5, "custom agent deleted")
with open(tmp_agents, encoding="utf-8") as f:
    check(json.load(f) == [], "deletion persisted to agents.json")
try:
    arena.delete_agent("warren")
    check(False, "deleting a default must fail")
except ValueError:
    check(True, "deleting a default refused")
try:
    arena.delete_agent("nope-123")
    check(False, "deleting unknown id must fail")
except KeyError:
    check(True, "deleting unknown id -> KeyError")

# ===========================================================================
print("\n[6] Arena — reset and season auto-restart")
# ===========================================================================
s_before = arena.get_state()["season"]
state = arena.reset()
check(state["season"] == s_before + 1, "manual reset bumps the season")
check(state["day_index"] == START_OFFSET, "reset rewinds the clock to the warm-up line")

# Auto-restart: pretend the season started long ago so the clock is past the end.
arena.started_at = time.time() - (len(sample) + 10) * arena.seconds_per_day
s_after = arena.get_state()["season"]  # this get_state triggers the rollover
check(s_after == state["season"] + 1, "season auto-restarts after the last day")
check(arena.get_state()["day_index"] >= START_OFFSET, "new season starts at the offset")

# ===========================================================================
print("\n[7] Custom backtest through the real API handler")
# ===========================================================================
from fastapi import HTTPException

from backend.main import BacktestRequest, backtest

req = BacktestRequest(ticker="BULLTEST", start="2023-01-01", end="2024-06-01",
                      strategy="custom",
                      params={"buy_rule": "SMA(10) > SMA(30)",
                              "sell_rule": "SMA(10) < SMA(30)"})
result = backtest(req)
check(result["strategy"] == "Custom Rules", "custom backtest labelled 'Custom Rules'")
check(len(result["equity"]) == len(result["dates"]), "equity aligned with dates")
check("metrics" in result and "sharpe" in result["metrics"], "metrics present")
check(result["equity"][0] == 10000.0, "starts at starting_cash")

# Bad custom rules -> HTTP 400 with the parser's message.
req.params = {"buy_rule": "WAT > 1", "sell_rule": "CLOSE < 0"}
try:
    backtest(req)
    check(False, "bad custom rules must 400")
except HTTPException as e:
    check(e.status_code == 400 and "WAT" in str(e.detail),
          f"400 with message: {str(e.detail)[:60]!r}")

# Built-in strategies still work untouched.
req2 = BacktestRequest(ticker="BULLTEST", start="2023-01-01", end="2024-06-01",
                       strategy="sma_crossover", params={"fast": 10, "slow": 30})
result2 = backtest(req2)
check(result2["strategy"] == "SMA Crossover", "built-in strategies unchanged")
# Same rules, same math: the custom DSL version must match the built-in.
check(result["metrics"]["final_equity"] == result2["metrics"]["final_equity"],
      "custom SMA(10)>SMA(30) == built-in sma_crossover(10,30)")

# ===========================================================================
print("\n[8] Bollinger Bands strategy")
# ===========================================================================
check("bollinger_bands" in STRATEGIES, "bollinger_bands registered in STRATEGIES")
bb_spec = STRATEGIES["bollinger_bands"]
for key in ("func", "label", "description", "params"):
    check(key in bb_spec, f"bollinger_bands spec has '{key}'")
param_names = {p["name"] for p in bb_spec["params"]}
check({"period", "num_std"} <= param_names, "bollinger_bands has period + num_std params")

# Small handmade DataFrame: a flat run, then a sharp drop (oversold), then a
# recovery back to/above the average. With period=3, num_std=1.
bb_closes = [100, 100, 100, 90, 100, 100, 100]
bb_idx = pd.bdate_range("2024-02-01", periods=len(bb_closes))
bb_df = pd.DataFrame({"Close": bb_closes}, index=bb_idx)
bb_pos = bollinger_bands(bb_df, period=3, num_std=1.0)

check(set(bb_pos.unique()) <= {0, 1}, "bollinger_bands positions are 0/1 only")
check(list(bb_pos.index) == list(bb_df.index), "bollinger_bands positions share the prices' index")
# First two days: rolling stats need 3 days of history -> NaN -> flat (0).
check(list(bb_pos[:2]) == [0, 0], "bollinger_bands stays flat during warm-up (no lookahead)")

# Day 3 (index 3, price 90): mean of [100,100,90]=96.67, std~5.77,
# lower = 96.67 - 5.77 = 90.9 -> 90 < 90.9 -> oversold -> BUY (position=1).
check(bb_pos.iloc[3] == 1, "bollinger_bands buys when price closes below the lower band")
# Day 4 (price 100): mean of [100,90,100]=96.67 -> 100 > 96.67 (middle) -> exit -> flat.
check(bb_pos.iloc[4] == 0, "bollinger_bands exits once price reverts above the middle band")

# No-lookahead check: position on day T must depend only on data up to day T.
# Truncating the series and recomputing must give the same positions for the
# overlapping days.
bb_pos_truncated = bollinger_bands(bb_df.iloc[:5], period=3, num_std=1.0)
check(list(bb_pos_truncated) == list(bb_pos.iloc[:5]),
      "bollinger_bands positions on day T unaffected by future data (no lookahead)")

# Runs through the real backtest engine without error.
bb_result = run_backtest(bb_df, bb_pos)
check("metrics" in bb_result, "bollinger_bands runs through run_backtest")

# ===========================================================================
print("\n[9] Sortino + Calmar metrics")
# ===========================================================================
sample2 = _generate_sample_data("METRICTEST", "2022-01-01", "2024-06-01")
normal_pos = pd.Series(1, index=sample2.index)  # buy & hold -> normal mixed returns
normal_result = run_backtest(sample2, normal_pos)
nm = normal_result["metrics"]

check("sortino" in nm, "sortino present in metrics")
check("calmar_ratio" in nm, "calmar_ratio present in metrics")
for key in ("sortino", "calmar_ratio"):
    val = nm[key]
    check(isinstance(val, (int, float)), f"{key} is numeric")
    check(val == val, f"{key} is not NaN")              # NaN != NaN
    check(val not in (float("inf"), float("-inf")), f"{key} is finite")

# Edge case: zero downside deviation. A strategy that's flat (always in
# cash) has zero daily returns every day -> no negative returns at all.
flat_pos = pd.Series(0, index=sample2.index)
flat_result = run_backtest(sample2, flat_pos)
fm = flat_result["metrics"]
check(fm["sortino"] == 0.0, "sortino handles zero-downside-deviation without crashing (returns 0.0)")
check(fm["calmar_ratio"] == 0.0, "calmar_ratio handles zero-drawdown without crashing (returns 0.0)")
check(fm["max_drawdown_pct"] == 0.0, "flat (all-cash) strategy has zero drawdown")

# ===========================================================================
print("\n[10] Optimizer — grid search (backend/analysis.py)")
# ===========================================================================
# A few years of synthetic data, plenty long for SMA(50)/SMA(100) to warm up.
opt_prices = _generate_sample_data("OPTTEST", "2018-01-01", "2024-01-01")

opt_grid = {"fast": [5, 10, 20], "slow": [50, 100]}
opt_out = optimize_strategy(opt_prices, "sma_crossover", opt_grid, "sharpe", 10_000.0)

check(opt_out["param_names"] == ["fast", "slow"], "optimize_strategy preserves param_names order")
check(len(opt_out["results"]) == 3 * 2, "optimize_strategy covers the full cartesian product (3x2=6)")
for r in opt_out["results"]:
    check(set(r["params"].keys()) == {"fast", "slow"}, "each result has both param names")
    check("sharpe" in r["metrics"] and "final_equity" in r["metrics"], "each result has full metrics dict")

# "best" really is the result with the highest sharpe among all results.
best_sharpe = max(r["metrics"]["sharpe"] for r in opt_out["results"])
check(opt_out["best"]["metrics"]["sharpe"] == best_sharpe, "best result has the max objective value")
check(opt_out["best"]["params"] in [r["params"] for r in opt_out["results"]],
      "best params are one of the grid combos")

# Safety limits: too many params, too many values, bad objective, unknown param.
try:
    optimize_strategy(opt_prices, "sma_crossover", {"fast": [5], "slow": [50], "period": [10]}, "sharpe", 10_000.0)
    check(False, "3-param grid should raise ValueError")
except ValueError as e:
    check("at most 2" in str(e), f"3-param grid rejected: {str(e)[:60]!r}")

try:
    optimize_strategy(opt_prices, "sma_crossover", {"fast": list(range(1, 14)), "slow": [50]}, "sharpe", 10_000.0)
    check(False, "13-value param should raise ValueError")
except ValueError as e:
    check("at most 12" in str(e), f"13-value param rejected: {str(e)[:60]!r}")

try:
    optimize_strategy(opt_prices, "sma_crossover", {"fast": [5, 10]}, "not_a_real_objective", 10_000.0)
    check(False, "bad objective should raise ValueError")
except ValueError as e:
    check("objective must be one of" in str(e), f"bad objective rejected: {str(e)[:60]!r}")

try:
    optimize_strategy(opt_prices, "sma_crossover", {"not_a_param": [1, 2]}, "sharpe", 10_000.0)
    check(False, "unknown param name should raise ValueError")
except ValueError as e:
    check("not a tunable parameter" in str(e), f"unknown param rejected: {str(e)[:60]!r}")

# ===========================================================================
print("\n[11] Walk-forward optimization (backend/analysis.py)")
# ===========================================================================
# Several years of data so each of the 5 segments (n_folds=4 -> 5 segments)
# is comfortably longer than SMA(100)'s warm-up.
wf_prices = _generate_sample_data("WFTEST", "2016-01-01", "2024-01-01")
wf_grid = {"fast": [5, 10], "slow": [50, 100]}

wf_out = walk_forward(wf_prices, "sma_crossover", wf_grid, "sharpe", 4, 10_000.0)

check(len(wf_out["folds"]) == 4, "walk_forward returns n_folds folds")
for fold in wf_out["folds"]:
    check(set(fold.keys()) >= {"train_range", "test_range", "best_params", "train_metrics", "test_metrics"},
          "each fold has train/test ranges, best_params, train/test metrics")
    check(len(fold["train_range"]) == 2 and len(fold["test_range"]) == 2,
          "train_range/test_range are [start, end] pairs")
    check(set(fold["best_params"].keys()) == {"fast", "slow"}, "fold's best_params cover the grid's params")
    check("sharpe" in fold["test_metrics"], "fold test_metrics has sharpe")

# Consecutive folds: fold i's test segment should immediately precede (or
# be adjacent to) fold i+1's train segment in time (rolling window).
check(wf_out["folds"][0]["test_range"][1] <= wf_out["folds"][1]["train_range"][1],
      "folds roll forward through time")

combined = wf_out["combined"]
check(len(combined["dates"]) == len(combined["equity"]) == len(combined["buy_hold"]) == len(combined["drawdown_pct"]),
      "combined out-of-sample curve arrays are all the same length")
check(combined["equity"][0] != 0, "combined equity curve starts from a real value")
check("sharpe" in combined["metrics"] and "final_equity" in combined["metrics"],
      "combined metrics dict has the same shape as run_backtest's metrics")

# Too many folds for the date range -> segments too short for SMA(100) -> ValueError.
short_prices = _generate_sample_data("WFSHORT", "2023-01-01", "2024-01-01")
try:
    walk_forward(short_prices, "sma_crossover", {"fast": [5, 10], "slow": [50, 100]}, "sharpe", 8, 10_000.0)
    check(False, "too many folds for short data should raise ValueError")
except ValueError as e:
    check("folds" in str(e).lower() or "rows" in str(e).lower(),
          f"too-short-segments rejected: {str(e)[:80]!r}")

# n_folds outside the allowed 2-8 range.
try:
    walk_forward(wf_prices, "sma_crossover", wf_grid, "sharpe", 1, 10_000.0)
    check(False, "n_folds=1 should raise ValueError")
except ValueError as e:
    check("n_folds" in str(e), f"n_folds=1 rejected: {str(e)[:60]!r}")

try:
    walk_forward(wf_prices, "sma_crossover", wf_grid, "sharpe", 9, 10_000.0)
    check(False, "n_folds=9 should raise ValueError")
except ValueError as e:
    check("n_folds" in str(e), f"n_folds=9 rejected: {str(e)[:60]!r}")

# ===========================================================================
print("\n[12] Monte Carlo bootstrap (backend/analysis.py)")
# ===========================================================================
mc_prices = _generate_sample_data("MCTEST", "2020-01-01", "2024-01-01")
mc_positions = sma_crossover(mc_prices, fast=10, slow=50)

mc_out = monte_carlo(mc_prices, mc_positions, 10_000.0, 500)

check(set(mc_out.keys()) == {"original", "percentiles", "bands", "prob_loss_pct", "prob_beat_buy_hold_pct"},
      "monte_carlo returns the documented top-level keys")
check(set(mc_out["percentiles"].keys()) == {"p5", "p25", "p50", "p75", "p95"},
      "monte_carlo percentiles has all 5 levels")

p = mc_out["percentiles"]
check(p["p5"]["final_equity"] <= p["p25"]["final_equity"] <= p["p50"]["final_equity"]
      <= p["p75"]["final_equity"] <= p["p95"]["final_equity"],
      "final_equity percentiles are non-decreasing p5<=p25<=p50<=p75<=p95")
check(p["p5"]["total_return_pct"] <= p["p50"]["total_return_pct"] <= p["p95"]["total_return_pct"],
      "total_return_pct percentiles are non-decreasing")

check(0 <= mc_out["prob_loss_pct"] <= 100, "prob_loss_pct is a valid percentage")
check(0 <= mc_out["prob_beat_buy_hold_pct"] <= 100, "prob_beat_buy_hold_pct is a valid percentage")

# Bands: one value per simulated trading day, for every percentile level.
n_days = len(mc_positions)
check(len(mc_out["bands"]["day"]) == n_days, "bands.day has one entry per trading day")
for key in ("p5", "p25", "p50", "p75", "p95"):
    check(len(mc_out["bands"][key]) == n_days, f"bands.{key} has one entry per trading day")
check(mc_out["bands"]["p5"][0] <= mc_out["bands"]["p95"][0],
      "bands p5 <= p95 even on day 0")

# "original" matches a normal run_backtest on the same positions.
mc_normal = run_backtest(mc_prices, mc_positions, starting_cash=10_000.0)
check(mc_out["original"]["final_equity"] == mc_normal["metrics"]["final_equity"],
      "monte_carlo's 'original' matches a plain run_backtest")

# Safety limits on n_sims.
try:
    monte_carlo(mc_prices, mc_positions, 10_000.0, 50)
    check(False, "n_sims=50 (<100) should raise ValueError")
except ValueError as e:
    check("n_sims" in str(e), f"n_sims=50 rejected: {str(e)[:60]!r}")

try:
    monte_carlo(mc_prices, mc_positions, 10_000.0, 5000)
    check(False, "n_sims=5000 (>2000) should raise ValueError")
except ValueError as e:
    check("n_sims" in str(e), f"n_sims=5000 rejected: {str(e)[:60]!r}")

# ===========================================================================
print("\n[13] Analysis API endpoints (/api/optimize, /api/walkforward, /api/montecarlo)")
# ===========================================================================
from backend.main import (
    MonteCarloRequest,
    OptimizeRequest,
    WalkForwardRequest,
    montecarlo,
    optimize,
    walkforward,
)

# --- /api/optimize ----------------------------------------------------------
opt_req = OptimizeRequest(
    ticker="APITEST", start="2018-01-01", end="2024-01-01",
    strategy="sma_crossover", param_grid={"fast": [5, 10], "slow": [50, 100]},
    objective="sharpe", starting_cash=10_000.0,
)
opt_resp = optimize(opt_req)
check(opt_resp["strategy"] == "SMA Crossover", "/api/optimize labels the strategy")
check(opt_resp["objective"] == "sharpe", "/api/optimize echoes the objective")
check(opt_resp["param_names"] == ["fast", "slow"], "/api/optimize echoes param_names")
check(len(opt_resp["results"]) == 4, "/api/optimize covers the full 2x2 grid")
check("best" in opt_resp and "params" in opt_resp["best"] and "metrics" in opt_resp["best"],
      "/api/optimize returns a 'best' with params + metrics")

# Bad param_grid (unknown strategy param) -> HTTP 400.
bad_opt_req = OptimizeRequest(
    ticker="APITEST", start="2018-01-01", end="2024-01-01",
    strategy="sma_crossover", param_grid={"not_a_param": [1, 2]},
    objective="sharpe", starting_cash=10_000.0,
)
try:
    optimize(bad_opt_req)
    check(False, "/api/optimize with bad param_grid should 400")
except HTTPException as e:
    check(e.status_code == 400, f"/api/optimize bad param_grid -> 400 ({str(e.detail)[:60]!r})")

# Unknown strategy -> HTTP 404.
unknown_opt_req = OptimizeRequest(
    ticker="APITEST", start="2018-01-01", end="2024-01-01",
    strategy="not_a_strategy", param_grid={"fast": [5, 10]},
    objective="sharpe", starting_cash=10_000.0,
)
try:
    optimize(unknown_opt_req)
    check(False, "/api/optimize with unknown strategy should 404")
except HTTPException as e:
    check(e.status_code == 404, f"/api/optimize unknown strategy -> 404 ({str(e.detail)[:60]!r})")

# --- /api/walkforward --------------------------------------------------------
wf_req = WalkForwardRequest(
    ticker="APIWF", start="2016-01-01", end="2024-01-01",
    strategy="sma_crossover", param_grid={"fast": [5, 10], "slow": [50, 100]},
    objective="sharpe", starting_cash=10_000.0, n_folds=4,
)
wf_resp = walkforward(wf_req)
check(wf_resp["strategy"] == "SMA Crossover", "/api/walkforward labels the strategy")
check(wf_resp["n_folds"] == 4, "/api/walkforward echoes n_folds")
check(len(wf_resp["folds"]) == 4, "/api/walkforward returns n_folds folds")
check("combined" in wf_resp and "dates" in wf_resp["combined"] and "metrics" in wf_resp["combined"],
      "/api/walkforward returns a combined out-of-sample curve + metrics")

# Bad request: too many folds for a short date range -> HTTP 400.
bad_wf_req = WalkForwardRequest(
    ticker="APIWFSHORT", start="2023-01-01", end="2024-01-01",
    strategy="sma_crossover", param_grid={"fast": [5, 10], "slow": [50, 100]},
    objective="sharpe", starting_cash=10_000.0, n_folds=8,
)
try:
    walkforward(bad_wf_req)
    check(False, "/api/walkforward with too-short segments should 400")
except HTTPException as e:
    check(e.status_code == 400, f"/api/walkforward too-short segments -> 400 ({str(e.detail)[:60]!r})")

# --- /api/montecarlo ----------------------------------------------------------
mc_req = MonteCarloRequest(
    ticker="APIMC", start="2020-01-01", end="2024-01-01",
    strategy="sma_crossover", params={"fast": 10, "slow": 50},
    starting_cash=10_000.0, n_sims=200,
)
mc_resp = montecarlo(mc_req)
check(mc_resp["strategy"] == "SMA Crossover", "/api/montecarlo labels built-in strategy")
check(mc_resp["n_sims"] == 200, "/api/montecarlo echoes n_sims")
check(set(mc_resp.keys()) >= {"original", "percentiles", "bands", "prob_loss_pct", "prob_beat_buy_hold_pct"},
      "/api/montecarlo returns the documented response shape")

# Custom DSL strategy works too.
mc_custom_req = MonteCarloRequest(
    ticker="APIMCCUSTOM", start="2020-01-01", end="2024-01-01",
    strategy="custom", params={"buy_rule": "SMA(10) > SMA(50)", "sell_rule": "SMA(10) < SMA(50)"},
    starting_cash=10_000.0, n_sims=150,
)
mc_custom_resp = montecarlo(mc_custom_req)
check(mc_custom_resp["strategy"] == "Custom Rules", "/api/montecarlo supports strategy='custom'")

# Bad request: n_sims out of range -> HTTP 400.
bad_mc_req = MonteCarloRequest(
    ticker="APIMC", start="2020-01-01", end="2024-01-01",
    strategy="sma_crossover", params={"fast": 10, "slow": 50},
    starting_cash=10_000.0, n_sims=10,
)
try:
    montecarlo(bad_mc_req)
    check(False, "/api/montecarlo with n_sims=10 should 400")
except HTTPException as e:
    check(e.status_code == 400, f"/api/montecarlo bad n_sims -> 400 ({str(e.detail)[:60]!r})")

# =====================================
# ===========================================================================
print("\n[14] Regime detector — feature computation")
# ===========================================================================
from backend.regime import (
    REGIMES,
    classify_regimes,
    compute_features,
    regime_mask,
)

# Use a long synthetic dataset so the warm-up is well past
regime_prices = _generate_sample_data("REGTEST", "2019-01-01", "2025-01-01")

# compute_features should return all 5 columns, no NaN after the warm-up period
regime_feat = compute_features(regime_prices)
check(set(regime_feat.columns) == {"vol_20", "trend_20", "autocorr_10", "hurst", "drawdown_20"},
      "compute_features returns all 5 expected columns")

# After the 40-day warm-up (longest single feature window = hurst at 40 days),
# all features should be finite (no NaN, no Inf).
feat_after_warmup = regime_feat.iloc[40:]
for col in regime_feat.columns:
    n_bad = feat_after_warmup[col].isna().sum() + np.isinf(feat_after_warmup[col]).sum()
    check(n_bad == 0, f"compute_features[{col}] has no NaN/Inf after warm-up")

# Sanity-check feature ranges (not strict, just catch obvious bugs)
check((feat_after_warmup["vol_20"] >= 0).all(), "vol_20 is non-negative (it's a std dev)")
check((feat_after_warmup["trend_20"] >= 0).all(), "trend_20 is non-negative (it's absolute slope)")
check((feat_after_warmup["autocorr_10"].between(-1.01, 1.01)).all(),
      "autocorr_10 is in [-1, 1]")
check((feat_after_warmup["hurst"].between(0.09, 0.91)).all(),
      "hurst is clamped to [0.1, 0.9]")
check((feat_after_warmup["drawdown_20"] <= 0.0001).all(),
      "drawdown_20 is <= 0 (fraction below the 20-day peak)")

# ===========================================================================
print("\n[15] Regime detector — classify_regimes output structure")
# ===========================================================================
regime_result = classify_regimes(regime_prices)

# Required top-level keys
required_keys = {"dates", "regimes", "scores", "features", "attribution_series",
                 "prices", "regime_counts", "latest"}
check(set(regime_result.keys()) >= required_keys,
      "classify_regimes returns all required top-level keys")

# Length consistency
n = len(regime_result["dates"])
check(n > 0, "classify_regimes returns at least one classified day")
check(len(regime_result["regimes"]) == n, "regimes length == dates length")
check(len(regime_result["scores"]) == n, "scores length == dates length")
check(len(regime_result["prices"]) == n, "prices length == dates length")
check(len(regime_result["attribution_series"]) == n, "attribution_series length == dates length")
for col in ["vol_20", "trend_20", "autocorr_10", "hurst", "drawdown_20"]:
    check(len(regime_result["features"][col]) == n,
          f"features[{col}] length == dates length")

# Warm-up days must be excluded (dates should start 100+ days into the price series)
first_date = pd.to_datetime(regime_result["dates"][0])
price_start = regime_prices.index[0]
check((first_date - price_start).days > 60,
      "first classified date is well after the price series start (warm-up excluded)")

# All regimes are valid strings
valid_regime_set = set(REGIMES)
check(all(r in valid_regime_set for r in regime_result["regimes"]),
      "all regime labels are from the valid set")

# Scores sum to ~1 per day (they're soft probabilities)
for i, score_row in enumerate(regime_result["scores"][:10]):  # check first 10
    total = sum(score_row.values())
    check(abs(total - 1.0) < 0.01, f"scores sum to ~1 on day {i} (got {total:.4f})")

# All four regimes appear (with a long enough dataset)
check(all(v > 0 for v in regime_result["regime_counts"].values()),
      "all four regimes appear at least once in a 6-year window")
check(sum(regime_result["regime_counts"].values()) == n,
      "regime_counts sum equals total classified days")

# Latest block has the right structure
latest = regime_result["latest"]
check(latest["regime"] in valid_regime_set, "latest.regime is a valid label")
check(abs(sum(latest["scores"].values()) - 1.0) < 0.01, "latest.scores sum to 1")
check(len(latest["attribution"]) == 3, "latest.attribution has exactly 3 entries")
for entry in latest["attribution"]:
    check(set(entry.keys()) >= {"feature", "z", "reason"},
          "each attribution entry has feature/z/reason")

# Attribution series: each entry should have 3 items with the right keys
check(all(len(a) == 3 for a in regime_result["attribution_series"][:5]),
      "each attribution_series entry has 3 items")
for entry_list in regime_result["attribution_series"][:3]:
    for entry in entry_list:
        check({"feature", "z", "reason"} <= set(entry.keys()),
              f"attribution entry {entry['feature']} has feature/z/reason")

# ===========================================================================
print("\n[16] Regime detector — regime_mask has shift(1) applied")
# ===========================================================================
mask = regime_mask(regime_prices, "TRENDING")

check(isinstance(mask, pd.Series), "regime_mask returns a pd.Series")
check(mask.dtype == bool, "regime_mask returns boolean dtype")
check(len(mask) == len(regime_prices), "regime_mask has same length as input prices")
check(list(mask.index) == list(regime_prices.index), "regime_mask shares prices' index")

# Shift(1) means day 0 of the mask must always be False
# (there's no "previous day's regime" for the very first price row)
check(mask.iloc[0] == False, "regime_mask day-0 is False (shift(1) applied, no prev day regime)")

# The mask should be True on some days (we have 6 years of data)
check(mask.any(), "regime_mask is True on at least some days")
# And False on some days (not all days are TRENDING)
check((~mask).any(), "regime_mask is False on at least some days")

# Test with all four valid regime labels
for r_label in REGIMES:
    m = regime_mask(regime_prices, r_label)
    check(isinstance(m, pd.Series) and m.dtype == bool,
          f"regime_mask({r_label!r}) returns a boolean Series")

# Invalid regime label should raise ValueError
try:
    regime_mask(regime_prices, "NOT_A_REGIME")
    check(False, "regime_mask with invalid label must raise ValueError")
except ValueError:
    check(True, "regime_mask with invalid label raises ValueError")

# ===========================================================================
print("\n[17] Regime filter in /api/backtest")
# ===========================================================================
from backend.main import BacktestRequest, RegimeRequest, backtest, regime as regime_endpoint

# --- /api/regime endpoint ---------------------------------------------------
r_req = RegimeRequest(ticker="REGTEST", start="2019-01-01", end="2025-01-01")
r_resp = regime_endpoint(r_req)
check(r_resp["ticker"] == "REGTEST", "/api/regime echoes ticker")
check(r_resp["data_source"] in {"yahoo", "cache", "sample"}, "/api/regime has data_source")
check("dates" in r_resp and "regimes" in r_resp and "scores" in r_resp,
      "/api/regime returns dates/regimes/scores")
check("latest" in r_resp and "regime_counts" in r_resp,
      "/api/regime returns latest + regime_counts")

# Too-short data -> HTTP 400
try:
    regime_endpoint(RegimeRequest(ticker="SHORT", start="2024-01-01", end="2024-06-01"))
    check(False, "/api/regime with too-short range should 400")
except HTTPException as e:
    check(e.status_code == 400, f"/api/regime too-short data -> 400 ({str(e.detail)[:60]!r})")

# --- regime_filter in /api/backtest ----------------------------------------
# Use a long date range so there's enough data for the regime warm-up
rf_req = BacktestRequest(
    ticker="RFTEST", start="2019-01-01", end="2025-01-01",
    strategy="sma_crossover", params={"fast": 10, "slow": 50},
    starting_cash=10_000.0, regime_filter="TRENDING",
)
rf_result = backtest(rf_req)

check("regime_filter" in rf_result, "regime_filter backtest result has 'regime_filter' key")
check(rf_result["regime_filter"] == "TRENDING", "regime_filter echoed in result")
check("regime_fraction_pct" in rf_result, "regime_filter backtest result has 'regime_fraction_pct'")
rf_frac = rf_result["regime_fraction_pct"]
check(isinstance(rf_frac, (int, float)) and 0 < rf_frac < 100,
      f"regime_fraction_pct is in (0, 100): got {rf_frac}")

# Un-filtered baseline exists and buy_hold is unchanged
rf_base = BacktestRequest(
    ticker="RFTEST", start="2019-01-01", end="2025-01-01",
    strategy="sma_crossover", params={"fast": 10, "slow": 50},
    starting_cash=10_000.0,
)
rf_base_result = backtest(rf_base)
check("regime_filter" not in rf_base_result,
      "unfiltered backtest does NOT include regime_filter key")
# buy_hold baseline is the same regardless of regime filter
check(rf_result["buy_hold"] == rf_base_result["buy_hold"],
      "buy_hold baseline is unaffected by regime_filter")

# Invalid regime_filter -> HTTP 400
try:
    backtest(BacktestRequest(
        ticker="RFTEST", start="2019-01-01", end="2025-01-01",
        strategy="sma_crossover", regime_filter="MADE_UP_REGIME",
    ))
    check(False, "invalid regime_filter should 400")
except HTTPException as e:
    check(e.status_code == 400, f"invalid regime_filter -> 400 ({str(e.detail)[:60]!r})")

# ===========================================================================
print("\n[18] Regime detector — edge cases")
# ===========================================================================
# Data too short for classify_regimes -> ValueError
short_prices = _generate_sample_data("ESHORT", "2024-01-01", "2024-04-01")
try:
    classify_regimes(short_prices)
    check(False, "classify_regimes with too-short data should raise ValueError")
except ValueError as e:
    check("100" in str(e) or "trading days" in str(e),
          f"short data ValueError has informative message: {str(e)[:80]!r}")

# With enough data, classify_regimes should never produce NaN features in output
regime_result2 = classify_regimes(regime_prices)
for col, values in regime_result2["features"].items():
    bad = sum(1 for v in values if v != v or v in (float("inf"), float("-inf")))
    check(bad == 0, f"output features[{col}] has no NaN/Inf")

# =====================================
print(f"\nALL {PASSED} CHECKS PASSED")
print("\n[19] GET /api/ticker/info — valid shape, graceful on bad ticker")
# ===========================================================================
from backend.main import ticker_info

# Any ticker (Yahoo may be blocked in the sandbox, so available may be False)
info_resp = ticker_info("AAPL")
check("ticker" in info_resp, "/api/ticker/info returns 'ticker' key")
check("available" in info_resp, "/api/ticker/info returns 'available' key")
check(isinstance(info_resp["available"], bool), "'available' is a bool")
check(info_resp["ticker"] == "AAPL", "'ticker' echoed correctly")

# When data is available, check the shape
if info_resp["available"]:
    for key in ("price", "prev_close", "change", "change_pct", "currency", "name"):
        check(key in info_resp, f"/api/ticker/info has '{key}' when available=true")
    check(isinstance(info_resp["price"], float), "price is a float")
    check(isinstance(info_resp["change_pct"], float), "change_pct is a float")
    check(isinstance(info_resp["currency"], str) and len(info_resp["currency"]) > 0,
          "currency is a non-empty string")
else:
    # Even without live data, the response must be well-formed (just available=false)
    check(True, "ticker_info returns available=false without crashing (no network)")

# Bad / nonsense ticker must also return available=false (not crash)
bad_info = ticker_info("THISISNOTAREALTICKER12345XYZ")
check(bad_info["ticker"] == "THISISNOTAREALTICKER12345XYZ",
      "bad ticker echoed in response")
check(bad_info["available"] == False, "bad ticker returns available=false, not a crash")
# Must NOT raise an HTTPException or any other exception (already caught above)

# Empty string ticker should not crash
empty_info = ticker_info("")
check("available" in empty_info, "empty ticker string returns a valid response shape")
check(empty_info["available"] == False, "empty ticker returns available=false")

# ===========================================================================
print("\n[20] POST /api/compare — multi-ticker comparison")
# ===========================================================================
from backend.main import CompareRequest, compare

# 2 synthetic-data tickers over the same date range
cmp_req = CompareRequest(
    tickers=["CMPTEST1", "CMPTEST2"],
    start="2020-01-01", end="2024-01-01",
    strategy="sma_crossover", params={"fast": 10, "slow": 50},
    starting_cash=10_000.0,
)
cmp_resp = compare(cmp_req)

check("dates" in cmp_resp, "/api/compare response has 'dates'")
check("tickers" in cmp_resp, "/api/compare response has 'tickers'")
check(isinstance(cmp_resp["dates"], list) and len(cmp_resp["dates"]) > 0,
      "/api/compare dates is a non-empty list")
check(len(cmp_resp["tickers"]) == 2, "/api/compare returns entry for each ticker")

for entry in cmp_resp["tickers"]:
    check("ticker" in entry, "each ticker entry has 'ticker'")
    if "error" not in entry:
        check("equity" in entry, "successful ticker entry has 'equity'")
        check("buy_hold" in entry, "successful ticker entry has 'buy_hold'")
        check("metrics" in entry, "successful ticker entry has 'metrics'")
        check("data_source" in entry, "successful ticker entry has 'data_source'")
        check(len(entry["equity"]) == len(cmp_resp["dates"]),
              "equity length matches dates length")
        check(len(entry["buy_hold"]) == len(cmp_resp["dates"]),
              "buy_hold length matches dates length")
        check("sharpe" in entry["metrics"], "metrics has 'sharpe'")

# Check that both tickers have the same equity length (aligned to common dates)
no_err = [e for e in cmp_resp["tickers"] if "error" not in e]
if len(no_err) >= 2:
    check(len(no_err[0]["equity"]) == len(no_err[1]["equity"]),
          "both ticker equity curves have the same length (common date range)")

# >6 tickers -> HTTP 400
try:
    compare(CompareRequest(
        tickers=["A", "B", "C", "D", "E", "F", "G"],
        start="2020-01-01", end="2024-01-01",
        strategy="sma_crossover",
    ))
    check(False, "/api/compare with >6 tickers should 400")
except HTTPException as e:
    check(e.status_code == 400, f"/api/compare >6 tickers -> 400 ({str(e.detail)[:60]!r})")

# Empty tickers list -> HTTP 400
try:
    compare(CompareRequest(tickers=[], start="2020-01-01", end="2024-01-01", strategy="sma_crossover"))
    check(False, "/api/compare with empty tickers list should 400")
except HTTPException as e:
    check(e.status_code == 400, f"/api/compare empty list -> 400 ({str(e.detail)[:60]!r})")

# Unknown strategy -> HTTP 404
try:
    compare(CompareRequest(
        tickers=["CMPTEST1"], start="2020-01-01", end="2024-01-01",
        strategy="not_a_real_strategy",
    ))
    check(False, "/api/compare with unknown strategy should 404")
except HTTPException as e:
    check(e.status_code == 404, f"/api/compare unknown strategy -> 404 ({str(e.detail)[:60]!r})")

# ===========================================================================
print("\n[21] POST /api/backtest — annual_returns + monthly_returns")
# ===========================================================================
# Use a multi-year range to get several annual/monthly buckets
amr_req = BacktestRequest(
    ticker="AMRTEST", start="2020-01-01", end="2024-01-01",
    strategy="sma_crossover", params={"fast": 10, "slow": 50},
    starting_cash=10_000.0,
)
amr_result = backtest(amr_req)

check("annual_returns" in amr_result, "/api/backtest result has 'annual_returns'")
check("monthly_returns" in amr_result, "/api/backtest result has 'monthly_returns'")

annual = amr_result["annual_returns"]
monthly = amr_result["monthly_returns"]

# Types: dict of str -> float
check(isinstance(annual, dict), "annual_returns is a dict")
check(isinstance(monthly, dict), "monthly_returns is a dict")
check(all(isinstance(k, str) for k in annual.keys()),
      "annual_returns keys are strings (years)")
check(all(isinstance(v, (int, float)) for v in annual.values()),
      "annual_returns values are numbers")

# Years must cover the full date range (2020–2023 for start=2020 end=2024)
expected_years = {"2020", "2021", "2022", "2023"}
check(expected_years <= set(annual.keys()),
      f"annual_returns covers all years in the date range: {sorted(annual.keys())}")

# Monthly returns: nested dict year -> month_num_str -> float
for yr, months in monthly.items():
    check(isinstance(months, dict), f"monthly_returns[{yr}] is a dict")
    check(all(isinstance(k, str) and k.isdigit() for k in months.keys()),
          f"monthly_returns[{yr}] keys are string month numbers")
    check(all(isinstance(v, (int, float)) for v in months.values()),
          f"monthly_returns[{yr}] values are numbers")
    month_ints = [int(k) for k in months.keys()]
    check(all(1 <= m <= 12 for m in month_ints),
          f"monthly_returns[{yr}] month numbers are 1-12")

# No NaN/Inf in either dict
for yr, v in annual.items():
    check(v == v and v not in (float("inf"), float("-inf")),
          f"annual_returns[{yr}] is finite")

for yr, months in monthly.items():
    for mo, v in months.items():
        check(v == v and v not in (float("inf"), float("-inf")),
              f"monthly_returns[{yr}][{mo}] is finite")

# ===========================================================================
print("\n[22] POST /api/regime/attribution")
# ===========================================================================
from backend.main import RegimeAttributionRequest, regime_attribution

attr_req = RegimeAttributionRequest(
    ticker="ATTRTEST", start="2019-01-01", end="2025-01-01",
    strategy="sma_crossover", params={"fast": 10, "slow": 50},
    starting_cash=10_000.0,
)
attr_resp = regime_attribution(attr_req)

# Top-level shape
check("strategy" in attr_resp, "/api/regime/attribution has 'strategy'")
check("attribution" in attr_resp, "/api/regime/attribution has 'attribution'")
check("overall_metrics" in attr_resp, "/api/regime/attribution has 'overall_metrics'")
check("unclassified_days" in attr_resp, "/api/regime/attribution has 'unclassified_days'")

# Attribution must contain all 4 regime keys
from backend.regime import REGIMES as ALL_REGIMES
attr_dict = attr_resp["attribution"]
for regime_lbl in ALL_REGIMES:
    check(regime_lbl in attr_dict,
          f"/api/regime/attribution has key '{regime_lbl}'")
    entry = attr_dict[regime_lbl]
    check("days" in entry, f"attribution[{regime_lbl}] has 'days'")
    check("days_pct" in entry, f"attribution[{regime_lbl}] has 'days_pct'")
    check("return_pct" in entry, f"attribution[{regime_lbl}] has 'return_pct'")
    check("ann_return_pct" in entry, f"attribution[{regime_lbl}] has 'ann_return_pct'")
    # Values are finite numbers
    check(isinstance(entry["return_pct"], (int, float)) and
          entry["return_pct"] == entry["return_pct"] and
          entry["return_pct"] not in (float("inf"), float("-inf")),
          f"attribution[{regime_lbl}].return_pct is finite")
    check(isinstance(entry["ann_return_pct"], (int, float)) and
          entry["ann_return_pct"] == entry["ann_return_pct"] and
          entry["ann_return_pct"] not in (float("inf"), float("-inf")),
          f"attribution[{regime_lbl}].ann_return_pct is finite")

# Sum of days_pct + unclassified / total <= 100% (within rounding error)
classified_days_sum = sum(attr_dict[r]["days"] for r in ALL_REGIMES)
unclassified = attr_resp["unclassified_days"]
total_days_attributed = classified_days_sum + unclassified

# The total attributed days should equal the full price series length (give or take)
# We just check the sum of days_pct is <= 100 (unclassified makes up the rest)
sum_days_pct = sum(attr_dict[r]["days_pct"] for r in ALL_REGIMES)
check(sum_days_pct <= 100.01,
      f"attribution days_pct values sum to <= 100% (got {sum_days_pct:.2f}%)")
check(sum_days_pct > 0.0, "at least some days are classified (sum_days_pct > 0)")
check(isinstance(unclassified, int) and unclassified >= 0,
      "unclassified_days is a non-negative integer")

# overall_metrics matches the shape of run_backtest's metrics
om = attr_resp["overall_metrics"]
for key in ("total_return_pct", "sharpe", "max_drawdown_pct", "final_equity"):
    check(key in om, f"overall_metrics has '{key}'")

# Too-short data (< 150 days) -> HTTP 400
try:
    regime_attribution(RegimeAttributionRequest(
        ticker="SHORTATTR", start="2024-01-01", end="2024-06-01",
        strategy="sma_crossover",
    ))
    check(False, "/api/regime/attribution with < 150 days should 400")
except HTTPException as e:
    check(e.status_code == 400,
          f"/api/regime/attribution too-short data -> 400 ({str(e.detail)[:60]!r})")

print(f"\nALL {PASSED} CHECKS PASSED")
