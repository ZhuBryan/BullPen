"""
main.py — the FastAPI web server.
"""

import math
import os
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.ai import AIError, translate as ai_translate
from backend.analysis import optimize_strategy, walk_forward, monte_carlo
from backend.arena import get_arena
from backend.backtest import run_backtest
from backend.data import get_prices
from backend.regime import REGIMES, classify_regimes, regime_mask
from backend.rules import rule_positions, validate_rules
from backend.strategies import STRATEGIES

app = FastAPI(title="Bullpen — Trading Arena & Backtester")

FRONTEND = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "index.html"
)


class BacktestRequest(BaseModel):
    ticker: str = "AAPL"
    start: str = "2020-01-01"
    end: str = "2025-01-01"
    strategy: str = "sma_crossover"
    params: dict = {}
    starting_cash: float = 10_000.0
    regime_filter: Optional[str] = None


class RegimeRequest(BaseModel):
    ticker: str = "SPY"
    start: str = "2020-01-01"
    end: str = "2025-01-01"


class RulesValidateRequest(BaseModel):
    buy_rule: str = ""
    sell_rule: str = ""


class AITranslateRequest(BaseModel):
    description: str = ""
    provider: str = "openai"       # openai | anthropic | gemini | local
    api_key: str = ""              # used in memory only, never stored or logged
    model: str = ""               # optional; sensible default per provider
    base_url: str = ""            # for local / OpenAI-compatible servers


class AgentCreateRequest(BaseModel):
    name: str = ""
    emoji: str = ""
    buy_rule: str = ""
    sell_rule: str = ""


class ArenaResetRequest(BaseModel):
    ticker: Optional[str] = None


class OptimizeRequest(BaseModel):
    ticker: str = "AAPL"
    start: str = "2020-01-01"
    end: str = "2025-01-01"
    strategy: str = "sma_crossover"
    param_grid: dict = {}
    objective: str = "sharpe"
    starting_cash: float = 10_000.0


class WalkForwardRequest(OptimizeRequest):
    n_folds: int = 4


class MonteCarloRequest(BaseModel):
    ticker: str = "AAPL"
    start: str = "2020-01-01"
    end: str = "2025-01-01"
    strategy: str = "sma_crossover"
    params: dict = {}
    starting_cash: float = 10_000.0
    n_sims: int = 500


class CompareRequest(BaseModel):
    tickers: List[str]
    start: str = "2020-01-01"
    end: str = "2025-01-01"
    strategy: str = "sma_crossover"
    params: dict = {}
    starting_cash: float = 10_000.0


@app.get("/")
def home():
    return FileResponse(FRONTEND)


@app.get("/api/strategies")
def list_strategies():
    """Tell the frontend what strategies exist and what knobs they have."""
    return [
        {
            "id": key,
            "label": s["label"],
            "description": s["description"],
            "params": s["params"],
        }
        for key, s in STRATEGIES.items()
    ]


def _compute_annual_monthly_returns(
    dates: list, equity: list, starting_cash: float
) -> tuple:
    """
    Slice an already-computed equity curve into annual and monthly return buckets.

    annual_returns:  {year_str: total_return_pct_for_that_year}
    monthly_returns: {year_str: {month_str: return_pct_for_that_month}}

    Finance note: "return for period P" = equity_at_end_of_P / equity_at_start_of_P - 1.
    The equity curve already has commission and position shifts baked in, so we are
    just slicing it no lookahead bias introduced here.
    """
    df = pd.DataFrame({"equity": equity}, index=pd.to_datetime(dates))

    annual = {}
    monthly = {}

    # Group by year
    for year, year_group in df.groupby(df.index.year):
        year_str = str(year)
        # Year-start equity: use the equity value just before this year began.
        # For the very first partial year, use starting_cash as the prior year-end.
        prev_years = df[df.index.year < year]
        if prev_years.empty:
            year_start_equity = starting_cash
        else:
            year_start_equity = float(prev_years["equity"].iloc[-1])
        year_end_equity = float(year_group["equity"].iloc[-1])
        annual[year_str] = round((year_end_equity / year_start_equity - 1) * 100, 2)

        # Group the year's days by month
        monthly[year_str] = {}
        for month, month_group in year_group.groupby(year_group.index.month):
            month_str = str(month)
            # Month-start equity: last trading day of the prior month (or year_start if Jan)
            prev_days = df[df.index < month_group.index[0]]
            if prev_days.empty:
                month_start_equity = starting_cash
            else:
                month_start_equity = float(prev_days["equity"].iloc[-1])
            month_end_equity = float(month_group["equity"].iloc[-1])
            monthly[year_str][month_str] = round(
                (month_end_equity / month_start_equity - 1) * 100, 2
            )

    return annual, monthly


def _run_strategy(req):
    """
    Shared helper: load prices, compute positions, run backtest.
    Returns (result_dict, label, source, prices, positions).
    Used by /api/backtest and /api/regime/attribution.
    """
    if req.strategy != "custom" and req.strategy not in STRATEGIES:
        raise HTTPException(404, f"Unknown strategy: {req.strategy}")

    if req.strategy == "custom":
        buy_rule = str(req.params.get("buy_rule", "") or "")
        sell_rule = str(req.params.get("sell_rule", "") or "")
        ok, err = validate_rules(buy_rule, sell_rule)
        if not ok:
            raise HTTPException(400, err)

    prices, source = get_prices(req.ticker, req.start, req.end)
    if len(prices) < 30:
        raise HTTPException(400, "Not enough data — try a longer date range.")

    if req.strategy == "custom":
        positions = rule_positions(prices, buy_rule, sell_rule)
        label = "Custom Rules"
    else:
        spec = STRATEGIES[req.strategy]
        allowed = {p["name"] for p in spec["params"]}
        clean_params = {}
        for name, value in req.params.items():
            if name in allowed:
                try:
                    clean_params[name] = float(value) if "." in str(value) else int(value)
                except (TypeError, ValueError):
                    raise HTTPException(400, f"Parameter '{name}' must be a number.")
        positions = spec["func"](prices, **clean_params)
        label = spec["label"]

    # Optional regime filter
    # If the user wants to only trade during a specific market regime, AND
    # the strategy's positions with a regime mask. Days outside the chosen
    # regime are forced to cash (position = 0).
    regime_fraction = None
    if req.regime_filter is not None:
        valid_regimes = {"TRENDING", "MEAN_REVERTING", "HIGH_VOLATILITY", "CHOPPY"}
        if req.regime_filter not in valid_regimes:
            raise HTTPException(
                400,
                f"Unknown regime_filter '{req.regime_filter}'. "
                f"Must be one of: {sorted(valid_regimes)}"
            )
        try:
            mask = regime_mask(prices, req.regime_filter)
        except ValueError as e:
            raise HTTPException(400, str(e))
        positions = positions & mask.reindex(positions.index).fillna(False)
        # What fraction of days were in this regime?
        regime_fraction = round(float(mask.reindex(positions.index).fillna(False).mean() * 100), 2)

    result = run_backtest(prices, positions, starting_cash=req.starting_cash)
    result["ticker"] = req.ticker.upper()
    result["strategy"] = label
    result["data_source"] = source

    if req.regime_filter is not None:
        result["regime_filter"] = req.regime_filter
        result["regime_fraction_pct"] = regime_fraction

    return result, label, source, prices, positions


@app.post("/api/backtest")
def backtest(req: BacktestRequest):
    """
    Run a strategy backtest on a single ticker and return the equity curve,
    metrics, trade log, and now also annual + monthly returns for the
    strategy equity curve.

    "custom" = user-written DSL rules (see backend/rules.py), parsed by a
    whitelisted parser - never eval()'d. Everything else is a built-in.
    """
    # _run_strategy handles strategy validation, price loading, regime filter,
    # and calling run_backtest. It returns the full result dict plus prices/positions.
    result, _label, _source, _prices, _positions = _run_strategy(req)

    # --- Add annual and monthly return breakdowns ----------------------------
    # These are computed by slicing the already-computed equity curve into
    # calendar-year and calendar-month buckets. No new trades, no lookahead.
    annual, monthly = _compute_annual_monthly_returns(
        result["dates"], result["equity"], req.starting_cash
    )
    result["annual_returns"] = annual
    result["monthly_returns"] = monthly

    return result


@app.post("/api/rules/validate")
def rules_validate(req: RulesValidateRequest):
    """Always HTTP 200 - the body says whether the rules parse."""
    ok, err = validate_rules(req.buy_rule, req.sell_rule)
    if ok:
        return {"valid": True}
    return {"valid": False, "error": err}


@app.post("/api/ai/translate")
def ai_translate_endpoint(req: AITranslateRequest):
    """
    Turn a plain-English strategy description into validated buy/sell rules.

    The API key is used only for this single request and is never stored or
    logged. The model's output is checked by the same whitelisted parser that
    powers custom strategies, so nothing unsafe can slip through.
    """
    try:
        result = ai_translate(
            description=req.description,
            provider=req.provider,
            api_key=req.api_key,
            model=req.model,
            base_url=req.base_url,
        )
        return {"ok": True, **result}
    except AIError as e:
        # 200 with ok:false keeps the frontend logic simple (like rules/validate).
        return {"ok": False, "error": str(e)}


@app.get("/api/arena/state")
def arena_state():
    """The whole arena: replay clock, dates revealed so far, leaderboard."""
    return get_arena().get_state()


@app.post("/api/arena/agents")
def arena_add_agent(req: AgentCreateRequest):
    """Add a custom fighter. Joins at the current day with $10,000."""
    try:
        return get_arena().add_agent(req.name, req.emoji, req.buy_rule, req.sell_rule)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/arena/agents/{agent_id}")
def arena_delete_agent(agent_id: str):
    """Remove a custom agent. Default personalities cannot be deleted."""
    try:
        get_arena().delete_agent(agent_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except KeyError:
        raise HTTPException(404, f"No agent with id '{agent_id}'.")
    return {"ok": True}


@app.post("/api/arena/reset")
def arena_reset(req: ArenaResetRequest):
    """Start a new season now, optionally switching to a different ticker."""
    try:
        return get_arena().reset(req.ticker)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/regime")
def regime(req: RegimeRequest):
    """
    Classify each trading day into one of four market regimes (TRENDING,
    MEAN_REVERTING, HIGH_VOLATILITY, CHOPPY) using rolling statistical
    features. No ML model pure rolling math, fully explainable.

    Requires at least 150 trading days of data.
    """
    prices, source = get_prices(req.ticker, req.start, req.end)

    if len(prices) < 150:
        raise HTTPException(
            400,
            f"Need at least 150 trading days for regime detection "
            f"(got {len(prices)}). Try a longer date range."
        )

    try:
        result = classify_regimes(prices)
    except ValueError as e:
        raise HTTPException(400, str(e))

    result["ticker"] = req.ticker.upper()
    result["data_source"] = source
    return result


def _load_prices_for_analysis(ticker: str, start: str, end: str):
    """Shared price-loading + minimum-length check."""
    prices, source = get_prices(ticker, start, end)
    if len(prices) < 30:
        raise HTTPException(400, "Not enough data — try a longer date range.")
    return prices, source


@app.post("/api/optimize")
def optimize(req: OptimizeRequest):
    if req.strategy not in STRATEGIES:
        raise HTTPException(404, f"Unknown strategy: {req.strategy}")
    prices, _source = _load_prices_for_analysis(req.ticker, req.start, req.end)
    try:
        outcome = optimize_strategy(
            prices, req.strategy, req.param_grid, req.objective, req.starting_cash
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {
        "strategy": STRATEGIES[req.strategy]["label"],
        "objective": req.objective,
        "param_names": outcome["param_names"],
        "results": outcome["results"],
        "best": outcome["best"],
    }


@app.post("/api/walkforward")
def walkforward(req: WalkForwardRequest):
    if req.strategy not in STRATEGIES:
        raise HTTPException(404, f"Unknown strategy: {req.strategy}")
    prices, _source = _load_prices_for_analysis(req.ticker, req.start, req.end)
    try:
        outcome = walk_forward(
            prices, req.strategy, req.param_grid, req.objective, req.n_folds, req.starting_cash
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {
        "strategy": STRATEGIES[req.strategy]["label"],
        "objective": req.objective,
        "n_folds": req.n_folds,
        "folds": outcome["folds"],
        "combined": outcome["combined"],
    }


@app.post("/api/montecarlo")
def montecarlo(req: MonteCarloRequest):
    if req.strategy != "custom" and req.strategy not in STRATEGIES:
        raise HTTPException(404, f"Unknown strategy: {req.strategy}")
    if req.strategy == "custom":
        buy_rule = str(req.params.get("buy_rule", "") or "")
        sell_rule = str(req.params.get("sell_rule", "") or "")
        ok, err = validate_rules(buy_rule, sell_rule)
        if not ok:
            raise HTTPException(400, err)
    prices, _source = _load_prices_for_analysis(req.ticker, req.start, req.end)
    if req.strategy == "custom":
        positions = rule_positions(prices, buy_rule, sell_rule)
        label = "Custom Rules"
    else:
        spec = STRATEGIES[req.strategy]
        allowed = {p["name"] for p in spec["params"]}
        clean_params = {}
        for name, value in req.params.items():
            if name in allowed:
                try:
                    clean_params[name] = float(value) if "." in str(value) else int(value)
                except (TypeError, ValueError):
                    raise HTTPException(400, f"Parameter '{name}' must be a number.")
        positions = spec["func"](prices, **clean_params)
        label = spec["label"]
    try:
        outcome = monte_carlo(prices, positions, req.starting_cash, req.n_sims)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {
        "strategy": label,
        "n_sims": req.n_sims,
        "starting_cash": req.starting_cash,
        "original": outcome["original"],
        "percentiles": outcome["percentiles"],
        "bands": outcome["bands"],
        "prob_loss_pct": outcome["prob_loss_pct"],
        "prob_beat_buy_hold_pct": outcome["prob_beat_buy_hold_pct"],
    }


# ---------------------------------------------------------------------------
# NEW ENDPOINT 1 GET /api/ticker/info
# ---------------------------------------------------------------------------

@app.get("/api/ticker/info")
def ticker_info(ticker: str = "AAPL"):
    """
    Return the latest available price data for a ticker using yfinance.

    Because the sandbox may block Yahoo Finance network calls, this endpoint
    NEVER raises HTTP 4xx/5xx for bad tickers or network issues. Instead it
    returns {"available": false} so the UI can gracefully show "no data."

    If data IS available, it returns current price, previous close, and the
    day's change both in dollars and percent the numbers you'd see on a
    financial data site.
    """
    ticker = ticker.upper().strip()
    try:
        import yfinance as yf  # imported here so the rest of the app works without it

        yf_ticker = yf.Ticker(ticker)

        # Prefer fast_info (yfinance >= 0.2): a lightweight dict of frequently-
        # needed fields fetched without downloading a full historical table.
        try:
            fi = yf_ticker.fast_info
            price = float(fi.last_price) if fi.last_price is not None else None
            prev_close = float(fi.previous_close) if fi.previous_close is not None else None
            currency = getattr(fi, "currency", "USD") or "USD"
            # yfinance doesn't put the long name in fast_info; fall back to info dict.
            name = yf_ticker.info.get("longName") or yf_ticker.info.get("shortName") or ticker
        except Exception:
            # If fast_info fails (older yfinance or blocked), try a tiny download.
            df = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
            if df is None or df.empty:
                return {"ticker": ticker, "available": False}

            # Flatten multi-index columns if present (newer yfinance).
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if "Close" not in df.columns or len(df) < 2:
                return {"ticker": ticker, "available": False}

            price = float(df["Close"].iloc[-1])
            prev_close = float(df["Close"].iloc[-2])
            currency = "USD"
            name = ticker

        # Sanity check: we need both price points to compute a day-change.
        if price is None or prev_close is None or prev_close == 0:
            return {"ticker": ticker, "available": False}

        change = round(price - prev_close, 4)
        change_pct = round((price / prev_close - 1) * 100, 4)

        return {
            "ticker": ticker,
            "available": True,
            "price": round(price, 4),
            "prev_close": round(prev_close, 4),
            "change": change,
            "change_pct": change_pct,
            "currency": currency,
            "name": name,
        }
    except Exception:
        # Catch-all: any unexpected error (import error, network timeout, bad ticker)
        # returns available=false rather than a 500.
        return {"ticker": ticker, "available": False}


# ---------------------------------------------------------------------------
# NEW ENDPOINT 2 POST /api/compare
# ---------------------------------------------------------------------------

@app.post("/api/compare")
def compare(req: CompareRequest):
    """
    Run the same strategy on multiple tickers and return their equity curves
    aligned to a common date axis so you can compare apples to apples.

    Finance note: we use the INNER JOIN of all tickers' date ranges only
    dates where every ticker has a price. This avoids spurious gaps when
    one ticker (e.g. a newly-listed ETF) starts later than the others.

    Up to 6 tickers allowed. Each ticker is processed independently, so one
    bad ticker doesn't crash the others it gets an "error" field instead.
    """
    tickers = [t.upper().strip() for t in req.tickers]

    if len(tickers) == 0:
        raise HTTPException(400, "Provide at least one ticker.")
    if len(tickers) > 6:
        raise HTTPException(400, "Maximum 6 tickers per comparison request.")

    # Validate strategy (same logic as /api/backtest)
    if req.strategy not in STRATEGIES:
        raise HTTPException(404, f"Unknown strategy: {req.strategy}")

    spec = STRATEGIES[req.strategy]
    allowed = {p["name"] for p in spec["params"]}
    clean_params = {}
    for name, value in req.params.items():
        if name in allowed:
            try:
                clean_params[name] = float(value) if "." in str(value) else int(value)
            except (TypeError, ValueError):
                raise HTTPException(400, f"Parameter '{name}' must be a number.")

    # --- Step 1: load all price series independently -------------------------
    ticker_data = []
    for t in tickers:
        try:
            prices, source = get_prices(t, req.start, req.end)
            if len(prices) < 30:
                ticker_data.append({"ticker": t, "error": "Fewer than 30 trading days of data."})
            else:
                ticker_data.append({"ticker": t, "prices": prices, "source": source})
        except Exception as exc:
            ticker_data.append({"ticker": t, "error": str(exc)})

    # --- Step 2: find the common date range (inner join) ---------------------
    # We need dates where EVERY successful ticker has data.
    valid_items = [item for item in ticker_data if "prices" in item]

    common_dates = None
    for item in valid_items:
        idx = item["prices"].index
        common_dates = idx if common_dates is None else common_dates.intersection(idx)

    # If no valid tickers at all, return what we have (all errors).
    if common_dates is None or len(common_dates) == 0:
        return {
            "dates": [],
            "tickers": [{"ticker": item["ticker"], "error": item.get("error", "No common dates.")}
                        for item in ticker_data],
        }

    # After trimming to common dates, check the 30-day minimum again.
    too_short = [item["ticker"] for item in valid_items
                 if len(item["prices"].reindex(common_dates).dropna()) < 30]

    # --- Step 3: run backtest for each valid ticker on the common date slice --
    result_tickers = []
    for item in ticker_data:
        t = item["ticker"]
        if "error" in item:
            result_tickers.append({"ticker": t, "error": item["error"]})
            continue
        if t in too_short:
            result_tickers.append({"ticker": t, "error": "Fewer than 30 days in common date range."})
            continue

        try:
            prices_trimmed = item["prices"].reindex(common_dates)
            positions = spec["func"](prices_trimmed, **clean_params)
            bt = run_backtest(prices_trimmed, positions, starting_cash=req.starting_cash)
            result_tickers.append({
                "ticker": t,
                "data_source": item["source"],
                "equity": bt["equity"],
                "buy_hold": bt["buy_hold"],
                "metrics": bt["metrics"],
            })
        except Exception as exc:
            result_tickers.append({"ticker": t, "error": str(exc)})

    return {
        "dates": [str(d.date()) for d in common_dates],
        "tickers": result_tickers,
    }


# ---------------------------------------------------------------------------
# NEW ENDPOINT 3 POST /api/regime/attribution
# ---------------------------------------------------------------------------

class RegimeAttributionRequest(BaseModel):
    ticker: str = "AAPL"
    start: str = "2020-01-01"
    end: str = "2025-01-01"
    strategy: str = "sma_crossover"
    params: dict = {}
    starting_cash: float = 10_000.0
    regime_filter: Optional[str] = None


@app.post("/api/regime/attribution")
def regime_attribution(req: RegimeAttributionRequest):
    """
    Break down a strategy's performance by market regime.

    How it works:
      1. Run the normal backtest to get the daily strategy returns.
      2. Run classify_regimes() on the same prices to get a regime label per day.
      3. For each of the 4 regimes, find all days classified as that regime
         (AFTER the regime warm-up window), compound the strategy's daily returns
         on those days, and convert to a total return %.
      4. Annualize each regime's return using (1+r)^(252/days) - 1.

    Finance note: we attribute the return ON regime days the regime is
    known at end-of-day T and we're measuring what happened on those days.
    This is NOT a trading signal (we don't shift by 1); it's retrospective
    attribution for analysis.

    Requires at least 150 trading days (same as /api/regime).
    """
    # Validate strategy
    if req.strategy != "custom" and req.strategy not in STRATEGIES:
        raise HTTPException(404, f"Unknown strategy: {req.strategy}")

    if req.strategy == "custom":
        buy_rule = str(req.params.get("buy_rule", "") or "")
        sell_rule = str(req.params.get("sell_rule", "") or "")
        ok, err = validate_rules(buy_rule, sell_rule)
        if not ok:
            raise HTTPException(400, err)

    prices, source = get_prices(req.ticker, req.start, req.end)

    if len(prices) < 150:
        raise HTTPException(
            400,
            f"Need at least 150 trading days for regime attribution "
            f"(got {len(prices)}). Try a longer date range."
        )

    # --- Run the strategy to get daily positions ----------------------------
    if req.strategy == "custom":
        positions = rule_positions(prices, buy_rule, sell_rule)
        label = "Custom Rules"
    else:
        spec = STRATEGIES[req.strategy]
        allowed = {p["name"] for p in spec["params"]}
        clean_params = {}
        for name, value in req.params.items():
            if name in allowed:
                try:
                    clean_params[name] = float(value) if "." in str(value) else int(value)
                except (TypeError, ValueError):
                    raise HTTPException(400, f"Parameter '{name}' must be a number.")
        positions = spec["func"](prices, **clean_params)
        label = spec["label"]

    # --- Run the full backtest (for overall metrics) ------------------------
    bt = run_backtest(prices, positions, starting_cash=req.starting_cash)

    # --- Rebuild the strategy's daily returns (same math as run_backtest) ---
    # We need per-day returns to attribute them to regimes.
    # Shift positions by 1 day (no-lookahead), compute daily P&L including commissions.
    close = prices["Close"]
    pos_shifted = positions.reindex(close.index).fillna(0).shift(1).fillna(0)
    stock_ret = close.pct_change().fillna(0)
    trades = pos_shifted.diff().abs().fillna(pos_shifted.iloc[0])
    strat_daily_ret = pos_shifted * stock_ret - trades * 0.001  # 0.1% commission

    # --- Classify regimes ---------------------------------------------------
    try:
        regime_result = classify_regimes(prices)
    except ValueError as e:
        raise HTTPException(400, str(e))

    # Build a Series mapping date -> regime label (only for classified days)
    classified_dates = pd.to_datetime(regime_result["dates"])
    classified_labels = regime_result["regimes"]
    regime_series = pd.Series(
        classified_labels, index=classified_dates, dtype=str
    )

    # Align to the strategy's date index
    regime_aligned = regime_series.reindex(strat_daily_ret.index)

    # Days not covered by classify_regimes are NaN (warm-up period)
    unclassified_days = int(regime_aligned.isna().sum())

    # --- Compute per-regime performance -------------------------------------
    attribution = {}

    for regime_label in REGIMES:
        # Mask: True only on days classified as this regime
        mask = regime_aligned == regime_label
        regime_days = int(mask.sum())

        if regime_days == 0:
            attribution[regime_label] = {
                "days": 0,
                "days_pct": 0.0,
                "return_pct": 0.0,
                "ann_return_pct": 0.0,
            }
            continue

        days_pct = round(regime_days / len(strat_daily_ret) * 100, 2)

        # Compound the strategy's daily returns on these regime days.
        # (1 + r1) * (1 + r2) * ... - 1  = total return for the period.
        regime_returns = strat_daily_ret[mask]
        total_return = float((1 + regime_returns).prod() - 1)
        total_return_pct = round(total_return * 100, 2)

        # Annualize: if you earned `total_return` over `regime_days` trading days,
        # what would that be at a 252-day-per-year pace?
        # Formula: (1 + r)^(252 / days) - 1
        if regime_days > 1:
            ann_return = (1 + total_return) ** (252 / regime_days) - 1
        else:
            ann_return = total_return  # single day do not annualize

        # Clamp extreme values to avoid returning inf for very short windows
        if not math.isfinite(ann_return):
            ann_return = 0.0

        attribution[regime_label] = {
            "days": regime_days,
            "days_pct": days_pct,
            "return_pct": total_return_pct,
            "ann_return_pct": round(ann_return * 100, 2),
        }

    return {
        "strategy": label,
        "ticker": req.ticker.upper(),
        "data_source": source,
        "attribution": attribution,
        "unclassified_days": unclassified_days,
        "overall_metrics": bt["metrics"],
    }
