"""
fills.py — what a trade actually costs.

The default backtest charges a flat commission and fills at the close. Real
execution costs more than that, and the gap is where most "profitable" paper
strategies die. Three costs, all estimated from OHLCV we already have:

  1. HALF-SPREAD. You buy at the ask and sell at the bid, so you pay half the
     spread each way. Daily bars do not contain the spread, so we proxy it from
     the day's range: quiet days have tight spreads, wild days do not.

  2. MARKET IMPACT. Your own order pushes the price against you, and it scales
     with the SQUARE ROOT of participation, not linearly. Trading 4x the size
     costs about 2x the impact per share. This is one of the most robust
     empirical findings in market microstructure.

         impact = alpha * daily_volatility * sqrt(shares / daily_volume)

  3. COMMISSION. Flat, already handled by the backtest engine.

Honest limitation: this is a STATISTICAL estimate from daily bars. It is not
the queue-position model in engine/replay.py, which uses real L2 data and
answers "would my specific order have filled". Use this for daily strategies,
use the engine for microstructure work. Do not claim the former is the latter.

Run `python backend/fills.py` for the self-check.
"""

import numpy as np
import pandas as pd

# Impact coefficient. Published estimates cluster around 0.5-1.0 for equities;
# 1.0 is the conservative end, which is the right direction to be wrong in.
DEFAULT_ALPHA = 1.0
DEFAULT_PARTICIPATION = 0.01   # trade 1% of a day's volume
MIN_HALF_SPREAD = 0.0001       # 1bp floor, even for the calmest mega-cap
MAX_COST = 0.05                # 5% sanity cap so a bad bar cannot delete an account


def estimate_costs(prices: pd.DataFrame, participation: float = DEFAULT_PARTICIPATION,
                   alpha: float = DEFAULT_ALPHA, vol_window: int = 20) -> pd.Series:
    """Fractional cost of ONE trade (one side) on each day.

    Returns a Series aligned to `prices.index`, e.g. 0.0012 = 12bp.
    """
    close = prices["Close"]

    # --- half-spread proxy from the day's range ---------------------------
    if "High" in prices.columns and "Low" in prices.columns:
        rng = (prices["High"] - prices["Low"]) / close.replace(0, np.nan)
        # A day's high-low range is roughly 4-5x the effective spread for
        # liquid names; 0.1 * range is a deliberately conservative half-spread.
        half_spread = (0.1 * rng).clip(lower=MIN_HALF_SPREAD)
    else:
        half_spread = pd.Series(MIN_HALF_SPREAD, index=close.index)

    # --- square-root market impact ---------------------------------------
    daily_vol = close.pct_change().rolling(vol_window).std()
    daily_vol = daily_vol.bfill().fillna(0.0)
    impact = alpha * daily_vol * np.sqrt(max(participation, 0.0))

    total = (half_spread + impact).clip(upper=MAX_COST)
    return total.fillna(MIN_HALF_SPREAD)


def summary(costs: pd.Series) -> dict:
    """Human-readable view, in basis points."""
    return {
        "mean_bps": round(float(costs.mean()) * 1e4, 2),
        "median_bps": round(float(costs.median()) * 1e4, 2),
        "worst_bps": round(float(costs.max()) * 1e4, 2),
    }


def demo():
    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    rng_ = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng_.normal(0, 0.01, 120))), index=idx)
    calm = pd.DataFrame({"Close": close, "High": close * 1.005, "Low": close * 0.995,
                         "Volume": 1e6}, index=idx)
    wild = pd.DataFrame({"Close": close, "High": close * 1.08, "Low": close * 0.92,
                         "Volume": 1e6}, index=idx)

    c_calm, c_wild = estimate_costs(calm), estimate_costs(wild)
    assert (c_calm > 0).all(), "cost must always be positive"
    assert c_wild.mean() > c_calm.mean(), "wider ranges must cost more"
    assert (c_calm >= MIN_HALF_SPREAD).all(), "floor not applied"

    # square-root scaling: 4x the size is ~2x the impact, not 4x
    vol_only = pd.DataFrame({"Close": close, "High": close, "Low": close,
                             "Volume": 1e6}, index=idx)
    small = estimate_costs(vol_only, participation=0.01).mean() - MIN_HALF_SPREAD
    big = estimate_costs(vol_only, participation=0.04).mean() - MIN_HALF_SPREAD
    assert small > 0, "impact term should be non-zero on a volatile series"
    ratio = big / small
    assert 1.9 < ratio < 2.1, f"sqrt law violated: 4x size gave {ratio:.2f}x impact"

    # bigger participation always costs more
    assert (estimate_costs(calm, participation=0.10).mean()
            > estimate_costs(calm, participation=0.001).mean())

    # missing High/Low must not crash
    bare = pd.DataFrame({"Close": close, "Volume": 1e6}, index=idx)
    assert estimate_costs(bare).notna().all(), "close-only frame should still work"

    # a pathological bar cannot produce an absurd cost
    broken = pd.DataFrame({"Close": close, "High": close * 50, "Low": close * 0.01,
                           "Volume": 1e6}, index=idx)
    assert (estimate_costs(broken) <= MAX_COST).all(), "cost cap not enforced"

    s = summary(c_calm)
    assert s["mean_bps"] > 0 and s["worst_bps"] >= s["median_bps"]
    print(f"ok — costs: calm {summary(c_calm)['mean_bps']}bps, "
          f"wild {summary(c_wild)['mean_bps']}bps, sqrt-law ratio {ratio:.2f}")


if __name__ == "__main__":
    demo()
