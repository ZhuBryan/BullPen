"""
analysis.py — "serious quant" tools built on top of backtest.py.

This module answers three questions a careful trader should always ask
*before* trusting a backtest:

  1. OPTIMIZE — "Which parameter values would have worked best?"
     A simple grid search: try every combination of parameters in
     `param_grid`, run a full backtest for each, and rank them by an
     objective (Sharpe ratio, CAGR, etc.).

  2. WALK-FORWARD — "...but wouldn't picking the 'best' params on the
     WHOLE history just be cheating?"
     Yes! If you tune your strategy on the same data you test it on, of
     course it looks great — you've fit the parameters to that exact
     history, including its lucky breaks. This is called OVERFITTING.
     Walk-forward avoids it by splitting time into chunks: tune on chunk 1,
     then test (untouched, "out of sample") on chunk 2; tune on chunk 2,
     test on chunk 3; and so on. The combined out-of-sample curve is a much
     more honest estimate of how the strategy would have actually performed,
     because at every point the parameters were chosen using ONLY data that
     was already in the past relative to the test period.

  3. MONTE CARLO — "How much of my result was luck?"
     History only happened once. The sequence of daily returns we observed
     is just ONE possible ordering of those days. Monte Carlo bootstrapping
     shuffles (resamples with replacement) the strategy's daily returns
     thousands of times to build a distribution of "what if the same days
     had happened in a different order / some days repeated and others
     didn't?" This shows how much of the headline number depends on a
     specific lucky (or unlucky) sequence, versus the strategy's underlying
     edge.

None of this predicts the future — it's about understanding the robustness
and uncertainty of a backtest you already ran.
"""

import itertools

import numpy as np
import pandas as pd

from backend.backtest import run_backtest, TRADING_DAYS_PER_YEAR
from backend.strategies import STRATEGIES

# ---------------------------------------------------------------------------
# Shared safety limits (see SPEC.md section 5). These keep grid searches and
# simulations small enough to run quickly on a free server with no GPU.
# ---------------------------------------------------------------------------
MAX_GRID_PARAMS = 2          # at most 2 params in a param_grid
MAX_GRID_VALUES_PER_PARAM = 12  # at most 12 values per param
MAX_GRID_COMBOS = 144        # 12 * 12 -- the hard ceiling on combinations

VALID_OBJECTIVES = {"sharpe", "sortino", "calmar_ratio", "cagr_pct", "total_return_pct"}

MIN_N_SIMS, MAX_N_SIMS = 100, 2000
MIN_N_FOLDS, MAX_N_FOLDS = 2, 8


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _validate_param_grid(param_grid: dict, strategy_id: str) -> list[str]:
    """
    Check the param_grid against the shared safety limits AND against the
    strategy's declared parameters (from STRATEGIES[strategy_id]["params"]).
    Returns the ordered list of param names (so callers can build the
    cartesian product in a stable order). Raises ValueError on problems.
    """
    if strategy_id not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_id}")

    if not param_grid:
        raise ValueError("param_grid must have at least one parameter.")

    if len(param_grid) > MAX_GRID_PARAMS:
        raise ValueError(
            f"param_grid has {len(param_grid)} parameters, but at most "
            f"{MAX_GRID_PARAMS} are allowed (to keep the grid search small)."
        )

    allowed = {p["name"] for p in STRATEGIES[strategy_id]["params"]}
    for name, values in param_grid.items():
        if name not in allowed:
            raise ValueError(
                f"'{name}' is not a tunable parameter of strategy "
                f"'{strategy_id}'. Allowed: {sorted(allowed)}."
            )
        if not isinstance(values, (list, tuple)) or len(values) == 0:
            raise ValueError(f"param_grid['{name}'] must be a non-empty list of values.")
        if len(values) > MAX_GRID_VALUES_PER_PARAM:
            raise ValueError(
                f"param_grid['{name}'] has {len(values)} values, but at most "
                f"{MAX_GRID_VALUES_PER_PARAM} are allowed per parameter."
            )

    total_combos = 1
    for values in param_grid.values():
        total_combos *= len(values)
    if total_combos > MAX_GRID_COMBOS:
        raise ValueError(
            f"param_grid would produce {total_combos} combinations, but at "
            f"most {MAX_GRID_COMBOS} are allowed. Use fewer parameters or "
            f"fewer values per parameter."
        )

    return list(param_grid.keys())


def _validate_objective(objective: str) -> str:
    if objective not in VALID_OBJECTIVES:
        raise ValueError(
            f"objective must be one of {sorted(VALID_OBJECTIVES)}, got '{objective}'."
        )
    return objective


def _validate_n_sims(n_sims: int) -> int:
    if not (MIN_N_SIMS <= n_sims <= MAX_N_SIMS):
        raise ValueError(f"n_sims must be between {MIN_N_SIMS} and {MAX_N_SIMS}, got {n_sims}.")
    return n_sims


def _validate_n_folds(n_folds: int) -> int:
    if not (MIN_N_FOLDS <= n_folds <= MAX_N_FOLDS):
        raise ValueError(f"n_folds must be between {MIN_N_FOLDS} and {MAX_N_FOLDS}, got {n_folds}.")
    return n_folds


def _longest_lookback(strategy_id: str, param_grid: dict | None = None) -> int:
    """
    Estimate the longest "warm-up" window the strategy needs, e.g. SMA(200)
    needs 200 days of history before it produces a real (non-NaN) signal.
    We take the MAX over every numeric param's default AND (if given) every
    value offered in param_grid, so a segment that's too short for ANY combo
    we might try gets flagged.
    """
    longest = 0
    for p in STRATEGIES[strategy_id]["params"]:
        candidates = [p.get("default", 0)]
        if param_grid and p["name"] in param_grid:
            candidates += list(param_grid[p["name"]])
        longest = max(longest, max(int(c) for c in candidates))
    return longest


# ---------------------------------------------------------------------------
# 1. Optimizer — plain grid search
# ---------------------------------------------------------------------------
def _param_combos(param_grid: dict) -> list[dict]:
    """Cartesian product of param_grid values, as a list of {name: value} dicts,
    in the natural nested order (first param varies slowest)."""
    names = list(param_grid.keys())
    value_lists = [param_grid[name] for name in names]
    combos = []
    for combo in itertools.product(*value_lists):
        combos.append(dict(zip(names, combo)))
    return combos


def optimize_strategy(
    prices: pd.DataFrame,
    strategy_id: str,
    param_grid: dict,
    objective: str = "sharpe",
    starting_cash: float = 10_000.0,
) -> dict:
    """
    Grid search: for every combination of parameters in param_grid, run the
    strategy over `prices` and score it by `objective` (a key in the
    run_backtest metrics dict, where higher is always better for the
    objectives we allow).

    Returns:
        {
          "param_names": [...],
          "results": [{"params": {...}, "metrics": {...}}, ...],
          "best":    {"params": {...}, "metrics": {...}},
        }

    Why this can mislead you on its own: the "best" combo is, by
    definition, whichever one happened to fit THIS price history best.
    That's exactly why walk_forward() exists below -- it re-runs this same
    search but only lets each combo "see" data that's in the past relative
    to where it's tested.
    """
    param_names = _validate_param_grid(param_grid, strategy_id)
    objective = _validate_objective(objective)

    spec = STRATEGIES[strategy_id]
    func = spec["func"]

    results = []
    for params in _param_combos(param_grid):
        # Cast to int/float matching the strategy's declared types isn't
        # strictly necessary here -- the strategy functions already call
        # int()/float() internally where needed (see strategies.py).
        positions = func(prices, **params)
        outcome = run_backtest(prices, positions, starting_cash=starting_cash)
        results.append({"params": params, "metrics": outcome["metrics"]})

    best = max(results, key=lambda r: r["metrics"][objective])

    return {
        "param_names": param_names,
        "results": results,
        "best": best,
    }


# ---------------------------------------------------------------------------
# 2. Walk-forward optimization
# ---------------------------------------------------------------------------
def _segment_bounds(n_rows: int, n_segments: int) -> list[tuple[int, int]]:
    """Split range(n_rows) into n_segments consecutive (almost-)equal chunks
    of integer row indices, covering the whole range with no gaps/overlaps."""
    bounds = []
    base = n_rows // n_segments
    remainder = n_rows % n_segments
    start = 0
    for i in range(n_segments):
        # Spread the remainder rows across the first `remainder` segments so
        # all segments are within 1 row of each other in size.
        size = base + (1 if i < remainder else 0)
        end = start + size
        bounds.append((start, end))
        start = end
    return bounds


def walk_forward(
    prices: pd.DataFrame,
    strategy_id: str,
    param_grid: dict,
    objective: str = "sharpe",
    n_folds: int = 4,
    starting_cash: float = 10_000.0,
) -> dict:
    """
    Rolling walk-forward optimization (SPEC.md section 5).

    THE CORE IDEA -- avoiding "hindsight bias":
      Split the date range into n_folds + 1 consecutive segments
      S0, S1, ..., Sn. For each fold i in 0..n_folds-1:
        - TRAIN on segment Si: grid-search param_grid, pick whichever combo
          maximizes `objective` -- this is exactly optimize_strategy() above,
          but applied only to this slice of history.
        - TEST those exact params on segment S(i+1), the segment
          immediately AFTER training -- data the parameter search never saw.
      This mimics how you'd actually trade: at the start of each period you
      only know the past, never the future, and you re-tune periodically.

    Stitching the out-of-sample curve:
      Each fold's test segment produces its own equity curve starting at
      `starting_cash`. To get ONE combined curve covering S1..Sn, we convert
      each fold's equity curve to DAILY RETURNS, concatenate those returns
      fold-by-fold (dropping the first day of each fold after the first,
      since day 0 of every backtest has return 0 by construction), and
      compound them continuously from a single starting_cash. This way fold
      2 "picks up where fold 1 left off" instead of restarting at
      starting_cash again.
    """
    param_names = _validate_param_grid(param_grid, strategy_id)
    objective = _validate_objective(objective)
    n_folds = _validate_n_folds(n_folds)

    n_segments = n_folds + 1
    n_rows = len(prices)

    bounds = _segment_bounds(n_rows, n_segments)

    # Make sure every segment has enough rows for the strategy's longest
    # lookback to actually warm up and produce a real signal. Otherwise the
    # "strategy" is just flat/NaN-driven for the whole segment, and the
    # grid search would be meaningless.
    lookback = _longest_lookback(strategy_id, param_grid)
    min_seg_len = min(end - start for start, end in bounds)
    # Need a few extra days beyond the bare minimum so there's at least some
    # non-warm-up data to actually trade on.
    required = lookback + 5
    if min_seg_len < required:
        raise ValueError(
            f"Each of the {n_segments} segments has only ~{min_seg_len} rows "
            f"of data, but '{STRATEGIES[strategy_id]['label']}' needs at "
            f"least {required} rows (longest lookback {lookback} + warm-up "
            f"margin) per segment. Use fewer folds (n_folds) or a longer "
            f"date range."
        )

    folds = []
    stitched_returns = []  # list of pd.Series of daily returns, to concatenate
    stitched_bh_returns = []  # same, for buy & hold (for the combined buy_hold line)
    stitched_dates = []  # matching dates, kept in sync with stitched_returns

    for i in range(n_folds):
        train_start, train_end = bounds[i]
        test_start, test_end = bounds[i + 1]

        train_prices = prices.iloc[train_start:train_end]
        test_prices = prices.iloc[test_start:test_end]

        # --- Train: grid search on this segment only -----------------------
        train_opt = optimize_strategy(
            train_prices, strategy_id, param_grid, objective, starting_cash
        )
        best_params = train_opt["best"]["params"]
        train_metrics = train_opt["best"]["metrics"]

        # --- Test: run those exact params on the NEXT segment --------------
        func = STRATEGIES[strategy_id]["func"]
        test_positions = func(test_prices, **best_params)
        test_outcome = run_backtest(test_prices, test_positions, starting_cash=starting_cash)
        test_metrics = test_outcome["metrics"]

        # Recover this fold's daily strategy & buy-hold returns from its
        # equity curve (equity[t]/equity[t-1] - 1), so we can stitch them.
        test_equity = pd.Series(test_outcome["equity"])
        test_bh = pd.Series(test_outcome["buy_hold"])
        fold_returns = test_equity.pct_change().fillna(0.0)
        fold_bh_returns = test_bh.pct_change().fillna(0.0)

        fold_dates = test_prices.index

        if i == 0:
            stitched_returns.append(fold_returns)
            stitched_bh_returns.append(fold_bh_returns)
            stitched_dates.append(fold_dates)
        else:
            # Drop day 0 of this fold (return is always 0 by construction --
            # it's the "start of a new equity curve" artifact, not a real
            # day-over-day move) so it doesn't create a fake flat day in the
            # combined curve. Drop the matching date too, so dates stay
            # aligned with the stitched returns/equity.
            stitched_returns.append(fold_returns.iloc[1:])
            stitched_bh_returns.append(fold_bh_returns.iloc[1:])
            stitched_dates.append(fold_dates[1:])

        folds.append({
            "train_range": [str(train_prices.index[0].date()), str(train_prices.index[-1].date())],
            "test_range": [str(test_prices.index[0].date()), str(test_prices.index[-1].date())],
            "best_params": best_params,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
        })

    # --- Combine the out-of-sample returns into one continuous curve -------
    combined_returns = pd.concat(stitched_returns, ignore_index=True)
    combined_bh_returns = pd.concat(stitched_bh_returns, ignore_index=True)

    # Dates for the combined curve: segments S1..Sn with the first day of
    # each fold after the first dropped (to match stitched_returns above).
    combined_dates = stitched_dates[0]
    for extra in stitched_dates[1:]:
        combined_dates = combined_dates.append(extra)

    combined_equity = starting_cash * (1 + combined_returns).cumprod()
    combined_bh = starting_cash * (1 + combined_bh_returns).cumprod()

    running_max = combined_equity.cummax()
    drawdown = combined_equity / running_max - 1

    # --- Metrics for the combined out-of-sample curve, same shape as
    #     run_backtest's metrics dict, computed the same way -----------------
    n_days = len(combined_returns)
    years = n_days / TRADING_DAYS_PER_YEAR
    total_return = combined_equity.iloc[-1] / starting_cash - 1
    bh_return = combined_bh.iloc[-1] / starting_cash - 1
    cagr = (combined_equity.iloc[-1] / starting_cash) ** (1 / years) - 1 if years > 0 else 0.0

    std = combined_returns.std()
    sharpe = (combined_returns.mean() / std) * np.sqrt(TRADING_DAYS_PER_YEAR) if std > 0 else 0.0

    downside = combined_returns[combined_returns < 0]
    downside_std = downside.std()
    if downside_std and downside_std > 0:
        sortino = (combined_returns.mean() / downside_std) * np.sqrt(TRADING_DAYS_PER_YEAR)
    else:
        sortino = 0.0

    max_drawdown = drawdown.min()
    if max_drawdown < 0:
        calmar = (cagr * 100) / abs(max_drawdown * 100)
    else:
        calmar = 0.0

    combined = {
        "dates": [str(d.date()) for d in combined_dates],
        "equity": [round(float(v), 2) for v in combined_equity],
        "buy_hold": [round(float(v), 2) for v in combined_bh],
        "drawdown_pct": [round(float(v) * 100, 2) for v in drawdown],
        "metrics": {
            "total_return_pct": round(total_return * 100, 2),
            "buy_hold_return_pct": round(bh_return * 100, 2),
            "cagr_pct": round(cagr * 100, 2),
            "sharpe": round(float(sharpe), 2),
            "sortino": round(float(sortino), 2),
            "max_drawdown_pct": round(float(max_drawdown) * 100, 2),
            "calmar_ratio": round(float(calmar), 2),
            "final_equity": round(float(combined_equity.iloc[-1]), 2),
        },
    }

    return {"folds": folds, "combined": combined}


# ---------------------------------------------------------------------------
# 3. Monte Carlo bootstrap
# ---------------------------------------------------------------------------
def monte_carlo(
    prices: pd.DataFrame,
    positions: pd.Series,
    starting_cash: float = 10_000.0,
    n_sims: int = 500,
) -> dict:
    """
    Bootstrap resampling of a strategy's daily returns (SPEC.md section 5).

    WHY THIS MATTERS:
      A backtest gives you ONE number for "total return" -- but that number
      came from ONE specific sequence of up-days and down-days. If the
      market had had a rough patch in March instead of June, the same
      strategy could have ended up very differently. Monte Carlo simulates
      many alternate "what if the days came in a different order /
      repeated" scenarios by RESAMPLING WITH REPLACEMENT from the actual
      daily returns the strategy produced. This does NOT predict the
      future -- it measures how sensitive the original result was to the
      luck of the draw, using only days that actually happened.

    Steps:
      1. Run the normal backtest (caller already did this) to get the
         strategy's daily returns (post-shift, post-commission) and the
         buy & hold return for comparison.
      2. n_sims times: draw len(daily_returns) days WITH REPLACEMENT from
         those daily returns, compound them into a new equity curve, and
         record final equity / total return % / max drawdown %.
      3. Summarize with percentiles (p5/p25/p50/p75/p95) of each stat, and
         the percentile equity-curve "bands" over time (a fan chart).
      4. prob_loss_pct: fraction of sims ending below starting_cash.
         prob_beat_buy_hold_pct: fraction of sims whose total return beat
         the ORIGINAL (actually observed) buy & hold return.
    """
    n_sims = _validate_n_sims(n_sims)

    outcome = run_backtest(prices, positions, starting_cash=starting_cash)

    # Recover the strategy's daily returns from its equity curve. equity[0]
    # is always starting_cash (return 0 on day 0 by construction).
    equity = pd.Series(outcome["equity"], dtype=float)
    daily_returns = equity.pct_change().fillna(0.0).to_numpy()
    n_days = len(daily_returns)

    original = {
        "final_equity": outcome["metrics"]["final_equity"],
        "total_return_pct": outcome["metrics"]["total_return_pct"],
        "max_drawdown_pct": outcome["metrics"]["max_drawdown_pct"],
    }
    original_bh_return_pct = outcome["metrics"]["buy_hold_return_pct"]

    rng = np.random.default_rng()

    # Pre-allocate result arrays: one row per simulation.
    final_equities = np.empty(n_sims)
    total_returns_pct = np.empty(n_sims)
    max_drawdowns_pct = np.empty(n_sims)
    # Equity paths for every sim, so we can compute percentile "bands" over
    # time (a fan chart: where might the equity curve have been on day t?).
    equity_paths = np.empty((n_sims, n_days))

    for sim in range(n_sims):
        # Bootstrap resample: pick n_days daily returns, WITH REPLACEMENT,
        # from the original sequence. Order is randomized -- "day 0" of a
        # simulation is just whichever real day got drawn first.
        sampled = rng.choice(daily_returns, size=n_days, replace=True)
        sim_equity = starting_cash * np.cumprod(1 + sampled)
        equity_paths[sim] = sim_equity

        final = sim_equity[-1]
        final_equities[sim] = final
        total_returns_pct[sim] = (final / starting_cash - 1) * 100

        running_max = np.maximum.accumulate(sim_equity)
        drawdown = sim_equity / running_max - 1
        max_drawdowns_pct[sim] = drawdown.min() * 100

    # --- Percentile summaries -----------------------------------------------
    pct_levels = {"p5": 5, "p25": 25, "p50": 50, "p75": 75, "p95": 95}
    percentiles = {}
    for key, q in pct_levels.items():
        percentiles[key] = {
            "final_equity": round(float(np.percentile(final_equities, q)), 2),
            "total_return_pct": round(float(np.percentile(total_returns_pct, q)), 2),
            "max_drawdown_pct": round(float(np.percentile(max_drawdowns_pct, q)), 2),
        }

    # --- Fan-chart bands: percentile equity value at each day index ---------
    bands = {"day": list(range(n_days))}
    for key, q in pct_levels.items():
        band = np.percentile(equity_paths, q, axis=0)
        bands[key] = [round(float(v), 2) for v in band]

    # --- Risk stats -----------------------------------------------------------
    prob_loss_pct = round(float(np.mean(final_equities < starting_cash) * 100), 1)
    prob_beat_buy_hold_pct = round(
        float(np.mean(total_returns_pct > original_bh_return_pct) * 100), 1
    )

    return {
        "original": original,
        "percentiles": percentiles,
        "bands": bands,
        "prob_loss_pct": prob_loss_pct,
        "prob_beat_buy_hold_pct": prob_beat_buy_hold_pct,
    }
