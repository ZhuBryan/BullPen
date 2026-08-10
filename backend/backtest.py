"""
backtest.py — the quant core.

A backtest answers: "If I had traded this strategy in the past, what would
have happened?"

How it works here (simple but correct):
  1. A strategy looks at price history and outputs a *position* for each day:
     1 = fully invested (long), 0 = in cash.
  2. CRITICAL: we shift positions forward by one day. A signal computed from
     today's close can only be traded *tomorrow*. Skipping this step is called
     "lookahead bias" — the #1 rookie mistake that makes fake-amazing results.
  3. Daily portfolio return = position * stock's daily return, minus
     commission on the days we trade.
  4. From the daily returns we build the equity curve and compute the metrics
     quants actually use: CAGR, Sharpe ratio, max drawdown, win rate.
"""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def run_backtest(
    prices: pd.DataFrame,
    positions: pd.Series,
    starting_cash: float = 10_000.0,
    commission_pct: float = 0.001,  # 0.1% per trade, a realistic friction
    realistic_fills: bool = False,
    participation: float = 0.01,
) -> dict:
    """
    prices:    DataFrame with a 'Close' column, indexed by date
    positions: Series of 0/1 (same index), produced by a strategy
    realistic_fills: also charge an estimated half-spread and square-root
        market impact per trade (see fills.py), on top of commission. Off by
        default so existing results stay comparable.
    Returns a dict with the equity curve, metrics, and trade log.
    """
    close = prices["Close"]

    # --- Step 1: avoid lookahead bias ---------------------------------
    # Signal from day T is only acted on at day T+1.
    pos = positions.reindex(close.index).fillna(0).shift(1).fillna(0)

    # --- Step 2: daily returns ----------------------------------------
    stock_ret = close.pct_change().fillna(0)
    trades = pos.diff().abs().fillna(pos.iloc[0])  # 1 on days we enter/exit
    cost_per_trade = commission_pct
    if realistic_fills:
        from backend.fills import estimate_costs
        cost_per_trade = commission_pct + estimate_costs(prices, participation)
    strat_ret = pos * stock_ret - trades * cost_per_trade

    # --- Step 3: equity curves ----------------------------------------
    equity = starting_cash * (1 + strat_ret).cumprod()
    buy_hold = starting_cash * (1 + stock_ret).cumprod()

    # --- Step 4: metrics ----------------------------------------------
    n_days = len(close)
    years = n_days / TRADING_DAYS_PER_YEAR

    total_return = equity.iloc[-1] / starting_cash - 1
    bh_return = buy_hold.iloc[-1] / starting_cash - 1
    cagr = (equity.iloc[-1] / starting_cash) ** (1 / years) - 1 if years > 0 else 0.0

    # Sharpe ratio: average excess return / volatility, annualized.
    # > 1 is good, > 2 is great, > 3 means check your code for bugs :)
    std = strat_ret.std()
    sharpe = (strat_ret.mean() / std) * np.sqrt(TRADING_DAYS_PER_YEAR) if std > 0 else 0.0

    # Sortino ratio: like Sharpe, but only punishes DOWNSIDE volatility.
    # Sharpe treats a big up day and a big down day as equally "risky" --
    # but investors don't actually mind upside swings! Sortino instead
    # divides by the standard deviation of just the negative daily returns
    # ("downside deviation"), so a strategy that has wild upside days but
    # calm downside days scores better here than under Sharpe.
    downside = strat_ret[strat_ret < 0]
    downside_std = downside.std()
    if downside_std and downside_std > 0:
        sortino = (strat_ret.mean() / downside_std) * np.sqrt(TRADING_DAYS_PER_YEAR)
    else:
        # No down days at all (or only one, so std is undefined) -- there's
        # no downside risk to measure. Same convention as Sharpe: 0.0.
        sortino = 0.0

    # Max drawdown: worst peak-to-trough loss. The "how much pain" metric.
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_drawdown = drawdown.min()

    # Calmar ratio: annualized return per unit of worst-case pain, i.e.
    # CAGR divided by the magnitude of the max drawdown. A Calmar of 2
    # means you earned 2% CAGR for every 1% of the deepest dip you had to
    # sit through. Higher = better return for the drawdown you endured.
    if max_drawdown < 0:
        calmar = (cagr * 100) / abs(max_drawdown * 100)
    else:
        # No drawdown at all -- nothing to divide by, so there's no
        # meaningful "pain-adjusted" return to report. Default to 0.0.
        calmar = 0.0

    # --- Step 5: trade log + win rate ----------------------------------
    trade_log, entry_price, entry_date = [], None, None
    for date, p, prev in zip(close.index, pos, pos.shift(1).fillna(0)):
        if p == 1 and prev == 0:  # entered
            entry_price, entry_date = close.loc[date], date
        elif p == 0 and prev == 1 and entry_price is not None:  # exited
            exit_price = close.loc[date]
            trade_log.append({
                "entry_date": str(entry_date.date()),
                "exit_date": str(date.date()),
                "entry_price": round(float(entry_price), 2),
                "exit_price": round(float(exit_price), 2),
                "return_pct": round((exit_price / entry_price - 1) * 100, 2),
            })
            entry_price = None
    # Position still open at the end? Count it as a trade too.
    if entry_price is not None:
        exit_price = close.iloc[-1]
        trade_log.append({
            "entry_date": str(entry_date.date()),
            "exit_date": "open",
            "entry_price": round(float(entry_price), 2),
            "exit_price": round(float(exit_price), 2),
            "return_pct": round((exit_price / entry_price - 1) * 100, 2),
        })

    wins = sum(1 for t in trade_log if t["return_pct"] > 0)
    win_rate = wins / len(trade_log) if trade_log else 0.0

    # --- Step 6: raw price candles for the chart ----------------------
    # The frontend draws a candlestick chart of the underlying. We hand it
    # Open/High/Low/Close (and Volume for the volume bars). If a data source
    # only gives Close, we fall back to a flat candle so nothing breaks.
    ohlc = _build_ohlc(prices)
    volume = (
        [int(v) if pd.notna(v) else 0 for v in prices["Volume"]]
        if "Volume" in prices.columns else [0] * len(close)
    )

    return {
        "dates": [str(d.date()) for d in equity.index],
        "equity": [round(float(v), 2) for v in equity],
        "buy_hold": [round(float(v), 2) for v in buy_hold],
        "drawdown_pct": [round(float(v) * 100, 2) for v in drawdown],
        "ohlc": ohlc,
        "volume": volume,
        "metrics": {
            "total_return_pct": round(total_return * 100, 2),
            "buy_hold_return_pct": round(bh_return * 100, 2),
            "cagr_pct": round(cagr * 100, 2),
            "sharpe": round(float(sharpe), 2),
            "sortino": round(float(sortino), 2),
            "max_drawdown_pct": round(float(max_drawdown) * 100, 2),
            "calmar_ratio": round(float(calmar), 2),
            "win_rate_pct": round(win_rate * 100, 1),
            "num_trades": len(trade_log),
            "final_equity": round(float(equity.iloc[-1]), 2),
        },
        "trades": trade_log[-50:],  # show at most the last 50 trades
    }


def _build_ohlc(prices: pd.DataFrame) -> list:
    """
    Turn a price DataFrame into a list of {o, h, l, c} dicts the candlestick
    chart can draw directly. If Open/High/Low are missing (some data sources
    only give Close), we synthesize a flat candle from Close so the chart still
    renders instead of crashing.
    """
    close = prices["Close"]
    has_full = all(col in prices.columns for col in ("Open", "High", "Low"))
    out = []  # one {o,h,l,c} dict per trading day
    for i in range(len(close)):
        c = float(close.iloc[i])
        if has_full:
            o = float(prices["Open"].iloc[i])
            h = float(prices["High"].iloc[i])
            l = float(prices["Low"].iloc[i])
        else:
            o = h = l = c
        out.append({
            "o": round(o, 2), "h": round(h, 2),
            "l": round(l, 2), "c": round(c, 2),
        })
    return out
