"""
strategies.py — the trading brains.

Every strategy is a function: (prices DataFrame, **params) -> positions Series
where positions[date] is 1 (hold the stock) or 0 (hold cash).

The engine (backtest.py) handles the "trade tomorrow, not today" shift,
so strategies can freely use today's close in their math.

STRATEGIES is the registry the API uses to list and run them.
"""

import pandas as pd


def buy_and_hold(prices: pd.DataFrame) -> pd.Series:
    """The benchmark: buy on day one, never sell."""
    return pd.Series(1, index=prices.index)


def sma_crossover(prices: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    """
    Classic trend-following. Hold the stock while the short-term average
    price is above the long-term average ("the trend is up").
    """
    close = prices["Close"]
    fast_ma = close.rolling(int(fast)).mean()
    slow_ma = close.rolling(int(slow)).mean()
    return (fast_ma > slow_ma).astype(int)


def rsi_mean_reversion(prices: pd.DataFrame, period: int = 14,
                       buy_below: int = 30, sell_above: int = 70) -> pd.Series:
    """
    Mean reversion: buy when the stock looks oversold (RSI low),
    sell when it looks overbought (RSI high).
    RSI = Relative Strength Index, a 0-100 "how stretched is this move" dial.
    """
    close = prices["Close"]
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(int(period)).mean()
    loss = (-delta.clip(upper=0)).rolling(int(period)).mean()
    rs = gain / loss.replace(0, 1e-9)
    rsi = 100 - 100 / (1 + rs)

    # Stateful logic: enter when RSI < buy_below, exit when RSI > sell_above,
    # otherwise keep doing whatever we were doing.
    positions, holding = [], 0
    for value in rsi:
        if pd.isna(value):
            positions.append(0)
            continue
        if holding == 0 and value < buy_below:
            holding = 1
        elif holding == 1 and value > sell_above:
            holding = 0
        positions.append(holding)
    return pd.Series(positions, index=prices.index)


def bollinger_bands(prices: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """
    Mean reversion using Bollinger Bands.

    Bollinger Bands are just "how far is the price from its recent average,
    measured in standard deviations of volatility?" We compute a rolling
    average (the middle band) and a rolling standard deviation, then draw
    an upper band and a lower band `num_std` standard deviations away.
    When price is near the bottom band, it's statistically "stretched" to
    the downside -- often (not always!) it snaps back toward the average.

    Entry: close below the LOWER band (price has fallen unusually far below
    its recent average -> looks oversold, expect a bounce back up).
    Exit:  close back above the MIDDLE band (the moving average) -- once
    price has reverted to "normal", we take our profit and step aside.

    This mirrors rsi_mean_reversion's stateful enter/exit loop: once we're
    holding, we stay long until the exit condition fires, regardless of
    what happens to the bands in between.
    """
    close = prices["Close"]
    middle = close.rolling(int(period)).mean()
    std = close.rolling(int(period)).std()
    lower = middle - num_std * std
    upper = middle + num_std * std  # noqa: F841 (kept for clarity / future use)

    # Stateful logic: enter when price closes below the lower band,
    # exit when price closes back above the middle band (the average),
    # otherwise keep doing whatever we were doing.
    positions, holding = [], 0
    for price, mid, low in zip(close, middle, lower):
        if pd.isna(mid) or pd.isna(low):
            # Bands haven't warmed up yet (not enough days of history).
            positions.append(0)
            continue
        if holding == 0 and price < low:
            holding = 1
        elif holding == 1 and price > mid:
            holding = 0
        positions.append(holding)
    return pd.Series(positions, index=prices.index)


def momentum(prices: pd.DataFrame, lookback: int = 60) -> pd.Series:
    """
    Momentum: hold the stock if it went UP over the last `lookback` days.
    "Winners keep winning" — one of the most studied effects in finance.
    """
    close = prices["Close"]
    past_return = close / close.shift(int(lookback)) - 1
    return (past_return > 0).astype(int)


# ---------------------------------------------------------------------------
# Registry: name -> (function, human description, parameter spec for the UI)
# ---------------------------------------------------------------------------
STRATEGIES = {
    "buy_and_hold": {
        "func": buy_and_hold,
        "label": "Buy & Hold",
        "description": "Buy on day one and never sell. The benchmark to beat.",
        "params": [],
    },
    "sma_crossover": {
        "func": sma_crossover,
        "label": "SMA Crossover",
        "description": "Hold while the fast moving average is above the slow one (trend following).",
        "params": [
            {"name": "fast", "label": "Fast MA (days)", "default": 20, "min": 2, "max": 200},
            {"name": "slow", "label": "Slow MA (days)", "default": 50, "min": 5, "max": 400},
        ],
    },
    "rsi_mean_reversion": {
        "func": rsi_mean_reversion,
        "label": "RSI Mean Reversion",
        "description": "Buy when oversold (RSI low), sell when overbought (RSI high).",
        "params": [
            {"name": "period", "label": "RSI period", "default": 14, "min": 2, "max": 100},
            {"name": "buy_below", "label": "Buy when RSI below", "default": 30, "min": 5, "max": 50},
            {"name": "sell_above", "label": "Sell when RSI above", "default": 70, "min": 50, "max": 95},
        ],
    },
    "bollinger_bands": {
        "func": bollinger_bands,
        "label": "Bollinger Bands",
        "description": "Buy when price dips below its lower volatility band (oversold), sell when it reverts back above the average.",
        "params": [
            {"name": "period", "label": "Lookback period (days)", "default": 20, "min": 5, "max": 100},
            {"name": "num_std", "label": "Band width (std devs)", "default": 2.0, "min": 0.5, "max": 4.0},
        ],
    },
    "momentum": {
        "func": momentum,
        "label": "Momentum",
        "description": "Hold the stock only if it rose over the lookback window.",
        "params": [
            {"name": "lookback", "label": "Lookback (days)", "default": 60, "min": 5, "max": 252},
        ],
    },
}
