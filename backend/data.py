"""
data.py — gets historical stock prices.

Order of preference:
  1. Local CSV cache (data/ folder) — so we don't re-download the same thing
  2. yfinance (Yahoo Finance) — totally free, no API key needed
  3. Synthetic "sample" data — realistic fake prices so the app still works
     offline. The UI tells you when you're looking at sample data.

Everything returns a pandas DataFrame indexed by date with columns:
  Open, High, Low, Close, Volume
"""

import hashlib
import re
import os
from datetime import datetime

import numpy as np
import pandas as pd

# The data/ folder lives next to the backend/ folder
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def _safe_part(value: str) -> str:
    """Reduce a caller-supplied string to characters that cannot escape a path.

    ticker/start/end all come from HTTP request bodies and all end up in a
    filename that gets read from and written to. Replacing only "/" and "\\" is
    not enough: ".." alone walks up a directory. Whitelist instead of blacklist.
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", str(value))
    cleaned = cleaned.replace("..", "_")
    return cleaned[:32] or "_"


def _cache_path(ticker: str, start: str, end: str) -> str:
    safe = _safe_part(ticker.upper())
    return os.path.join(DATA_DIR, f"{safe}_{_safe_part(start)}_{_safe_part(end)}.csv")


def _download_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Try to download real data from Yahoo Finance. Returns None on failure."""
    try:
        import yfinance as yf

        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None

        # Newer yfinance versions return multi-level columns like ('Close', 'AAPL').
        # Flatten them down to plain names: Close, Open, ...
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "Date"
        return df.dropna()
    except Exception:
        return None


def _generate_sample_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Generate realistic fake prices using a "geometric random walk" —
    the standard simple model for stock prices in quant finance.
    Seeded by the ticker name so the same ticker always gives the same chart.
    """
    seed = int(hashlib.md5(ticker.upper().encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range(start=start, end=end)  # business days only
    n = len(dates)
    if n < 2:
        raise ValueError("Date range too short")

    # Daily drift ~8%/year, volatility ~25%/year (typical for a single stock)
    mu, sigma = 0.08 / 252, 0.25 / np.sqrt(252)
    daily_returns = rng.normal(mu, sigma, n)

    start_price = 20 + (seed % 400)  # ticker-dependent starting price
    close = start_price * np.exp(np.cumsum(daily_returns))

    # Build plausible OHLC around the close
    spread = np.abs(rng.normal(0, sigma, n)) * close
    open_ = np.roll(close, 1) * (1 + rng.normal(0, sigma / 2, n))
    open_[0] = start_price
    high = np.maximum(open_, close) + spread / 2
    low = np.minimum(open_, close) - spread / 2
    volume = rng.integers(1_000_000, 50_000_000, n)

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    df.index.name = "Date"
    return df


def get_prices(ticker: str, start: str, end: str) -> tuple[pd.DataFrame, str]:
    """
    Main entry point. Returns (dataframe, source) where source is one of
    'cache', 'yahoo', or 'sample'.
    """
    ticker = ticker.upper().strip()
    path = _cache_path(ticker, start, end)

    # 1. Cache
    if os.path.exists(path):
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        if not df.empty:
            return df, "cache"

    # 2. Real data from Yahoo (free). BULLPEN_OFFLINE=1 skips it, so CI runs
    # on the deterministic synthetic data instead of failing whenever Yahoo
    # is slow or rate-limits a shared runner.
    offline = os.environ.get("BULLPEN_OFFLINE") == "1"
    df = None if offline else _download_yfinance(ticker, start, end)
    if df is not None and len(df) >= 30:
        df.to_csv(path)  # cache for next time
        return df, "yahoo"

    # 3. Offline fallback
    return _generate_sample_data(ticker, start, end), "sample"
