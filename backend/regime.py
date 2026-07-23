"""
regime.py — market regime detector.

Classifies each trading day into one of four regimes based on rolling
statistical features of price behavior. No ML model, no lookahead — just
pure rolling math that any beginner can understand and verify.

The four regimes:
  TRENDING       — Prices have a clear directional bias; momentum strategies tend to work.
  MEAN_REVERTING — Prices oscillate around a level; dip-buying tends to work.
  HIGH_VOLATILITY— Unusually large daily swings regardless of direction.
  CHOPPY         — No clear trend, no reversion signal, low predictability.

How the classification works (plain English):
  1. For each day T, compute 5 rolling features using ONLY data up to T.
  2. Normalize each feature with a rolling z-score (100-day lookback) so
     a score of +1 means "one standard deviation above the recent average."
  3. Multiply each z-score by regime-specific weights to get a raw affinity
     score for each regime — like voting: "how strongly does today's data
     look like TRENDING vs. CHOPPY?"
  4. Clamp negatives to 0 and normalize to sum to 1 (soft probabilities).
  5. Dominant regime = argmax.  Also report top-3 feature drivers.
"""

import numpy as np
import pandas as np_unused  # not used directly, but signals intent
import pandas as pd

# ---------------------------------------------------------------------------
# Four valid regime labels
# ---------------------------------------------------------------------------
REGIMES = ["TRENDING", "MEAN_REVERTING", "HIGH_VOLATILITY", "CHOPPY"]

# ---------------------------------------------------------------------------
# Affinity weights — domain knowledge, not learned.
#
# Finance intuition behind each weight:
#
#   TRENDING: autocorrelation at lag-1 is the strongest signal — if yesterday
#   went up and today went up again, that's autocorrelation → momentum → trend.
#   Hurst > 0.5 means the series "remembers" its past direction (persistent).
#   Trend slope (OLS) confirms directional movement. A large drawdown would
#   contradict trending (hence negative).
#
#   MEAN_REVERTING: the opposite — negative autocorrelation (today reverses
#   yesterday), Hurst < 0.5 (anti-persistent = prices snap back), flat OLS
#   slope. A high drawdown is consistent with mean reversion (prices stretched).
#
#   HIGH_VOLATILITY: vol_20 dominates. A large drawdown is also a volatility
#   signal (fast moves down). Some directional trend can accompany vol spikes
#   (panic selling is also directional), so trend_20_z gets a small positive.
#
#   CHOPPY: low everything — low absolute trend slope, low volatility, low
#   absolute autocorrelation. We use absolute-value z-scores here because we
#   want to detect "near zero" not "positive" or "negative".
# ---------------------------------------------------------------------------
REGIME_WEIGHTS = {
    "TRENDING": {
        "autocorr_10_z":    +0.35,   # positive lag-1 autocorr = momentum = trend
        "hurst_z":          +0.30,   # Hurst > 0.5 = persistent (trending) series
        "trend_20_z":       +0.25,   # strong OLS slope = directional price move
        "neg_drawdown_20_z":+0.10,   # small drawdown is consistent with uptrend
    },
    "MEAN_REVERTING": {
        "neg_autocorr_10_z":+0.40,   # negative autocorr = prices reverse
        "neg_hurst_z":      +0.30,   # Hurst < 0.5 = anti-persistent = mean reversion
        "neg_trend_20_z":   +0.20,   # flat slope = no directional bias
        "drawdown_20_z":    +0.10,   # stretched down = potential reversion candidate
    },
    "HIGH_VOLATILITY": {
        "vol_20_z":         +0.70,   # annualized vol is the primary driver
        "neg_drawdown_20_z":+0.20,   # large drawdown often accompanies vol spikes
        "trend_20_z":       +0.10,   # trend can accompany vol (panic = fast directional move)
    },
    "CHOPPY": {
        "neg_trend_20_abs_z":+0.40,  # near-zero slope = no trend
        "neg_vol_20_z":     +0.30,   # near-zero vol = quiet = choppy
        "neg_autocorr_abs_z":+0.30,  # near-zero autocorr = no serial dependence
    },
}


# ---------------------------------------------------------------------------
# Feature computation helpers
# ---------------------------------------------------------------------------

def _hurst_window(log_price_window: np.ndarray) -> float:
    """
    Estimate Hurst exponent for a 40-day window of log-prices.

    Method: variance-time scaling.
    - For each lag tau, compute the variance of (log_price[t] - log_price[t-tau]).
    - If the series is random (H=0.5), variance scales linearly with tau.
    - If H>0.5 (trending), variance grows faster than linear.
    - If H<0.5 (mean-reverting), variance grows slower.
    - Fit log(variance) vs log(tau) with OLS; slope ≈ 2*H.

    Clamp result to [0.1, 0.9] to avoid degenerate extremes.
    """
    if len(log_price_window) < 20:
        return 0.5

    lags = [2, 4, 8, 16]
    log_vars = []
    log_taus = []

    for tau in lags:
        if tau >= len(log_price_window):
            continue
        diffs = log_price_window[tau:] - log_price_window[:-tau]
        var = np.var(diffs)
        if var <= 0:
            # Flat prices → no variance at any lag → undefined Hurst
            return 0.5
        log_vars.append(np.log(var))
        log_taus.append(np.log(tau))

    if len(log_taus) < 2:
        return 0.5

    # OLS fit: log(var) = slope * log(tau) + intercept
    # slope ≈ 2*H, so H ≈ slope / 2
    coeffs = np.polyfit(log_taus, log_vars, 1)
    hurst = coeffs[0] / 2.0

    # Clamp to [0.1, 0.9] — values outside this range usually mean very short
    # windows or degenerate data; the extremes are not meaningful.
    return float(np.clip(hurst, 0.1, 0.9))


def _autocorr_window(returns_window: np.ndarray) -> float:
    """
    Pearson lag-1 autocorrelation of daily returns over a 10-day window.

    autocorr > 0: today's move tends to continue tomorrow (momentum).
    autocorr < 0: today's move tends to reverse tomorrow (mean-reversion).
    autocorr ≈ 0: no serial dependence (random walk).
    """
    if len(returns_window) < 3:
        return 0.0
    r = returns_window
    # Split into r[:-1] (today) and r[1:] (tomorrow) pairs
    r_t = r[:-1]
    r_t1 = r[1:]
    if np.std(r_t) == 0 or np.std(r_t1) == 0:
        return 0.0  # no variance → correlation undefined, default to 0
    corr_matrix = np.corrcoef(r_t, r_t1)
    result = corr_matrix[0, 1]
    return 0.0 if np.isnan(result) else float(result)


def compute_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all five rolling features from the Close price series.
    No lookahead: every calculation at day T uses only data up to and including T.

    Returns a DataFrame indexed by date with columns:
      vol_20, trend_20, autocorr_10, hurst, drawdown_20
    """
    close = prices["Close"]
    log_ret = np.log(close / close.shift(1))   # daily log-returns
    log_price = np.log(close)

    # --- Feature 1: vol_20 ---------------------------------------------------
    # 20-day rolling standard deviation of log-returns, annualized by sqrt(252).
    # This is the most common measure of short-term volatility in quant finance.
    vol_20 = log_ret.rolling(20).std() * np.sqrt(252)

    # --- Feature 2: trend_20 -------------------------------------------------
    # OLS slope of log-prices over the past 20 days, divided by the last
    # log-price to make it a "daily % move" (scale-invariant).
    # We use abs() because we only care about strength, not direction, for the
    # TRENDING/CHOPPY detection — direction is captured by autocorr.
    def _trend_slope(window):
        n = len(window)
        if n < 2:
            return np.nan
        x = np.arange(n, dtype=float)
        coeffs = np.polyfit(x, window, 1)  # coeffs[0] = slope
        # Normalize: divide slope by last log-price value to get daily % terms
        last_val = window[-1]
        if last_val == 0:
            return 0.0
        return abs(coeffs[0] / last_val)

    trend_20 = log_price.rolling(20).apply(_trend_slope, raw=True)

    # --- Feature 3: autocorr_10 ----------------------------------------------
    # Lag-1 Pearson autocorrelation of daily returns over a 10-day rolling window.
    autocorr_10 = log_ret.rolling(10).apply(
        lambda w: _autocorr_window(w), raw=True
    )

    # --- Feature 4: hurst ----------------------------------------------------
    # Hurst exponent via variance-time method, 40-day rolling window.
    # H > 0.5 → trending, H ≈ 0.5 → random walk, H < 0.5 → mean-reverting.
    hurst = log_price.rolling(40).apply(
        lambda w: _hurst_window(w), raw=True
    )

    # --- Feature 5: drawdown_20 ----------------------------------------------
    # How far is today's price below the highest price in the last 20 days?
    # Always negative (or zero). Example: -0.05 means 5% below the recent peak.
    rolling_max_20 = close.rolling(20).max()
    drawdown_20 = close / rolling_max_20 - 1.0

    return pd.DataFrame({
        "vol_20": vol_20,
        "trend_20": trend_20,
        "autocorr_10": autocorr_10,
        "hurst": hurst,
        "drawdown_20": drawdown_20,
    }, index=close.index)


def _rolling_zscore(series: pd.Series, window: int = 100) -> pd.Series:
    """
    Rolling z-score: (value - rolling mean) / rolling std.
    Uses only past data (no lookahead). Where std is 0 or NaN, returns 0.
    """
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    # Where std is 0 or NaN, there's no meaningful spread, so z-score = 0
    z = (series - mean) / std
    return z.fillna(0.0).replace([np.inf, -np.inf], 0.0)


def _compute_regime_scores(feat_z: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame of z-scored features, compute the affinity score for
    each of the four regimes using the fixed REGIME_WEIGHTS table.

    Returns a DataFrame with columns [TRENDING, MEAN_REVERTING, HIGH_VOLATILITY, CHOPPY],
    each normalized to sum to 1 across regimes (soft probabilities).
    """
    n = len(feat_z)

    # Build derived z-score columns used in the affinity matrix
    # For TRENDING / MEAN_REVERTING: use signed z-scores
    # For CHOPPY: use z-scores of the absolute-value features (detect "near zero")
    ac_z = feat_z["autocorr_10_z"]
    h_z = feat_z["hurst_z"]
    tr_z = feat_z["trend_20_z"]
    v_z = feat_z["vol_20_z"]
    dd_z = feat_z["drawdown_20_z"]

    # Absolute-value z-scores for CHOPPY (near-zero detectors)
    # We want to know if |trend| and |autocorr| are unusually low
    # by z-scoring the raw feature first, then taking abs of the z-score
    # is NOT what we want. Instead, z-score of abs(feature).
    trend_abs_z = feat_z["trend_20_abs_z"]
    autocorr_abs_z = feat_z["autocorr_10_abs_z"]

    # Affinity scores (raw, before clamping)
    trending = (
        ac_z * 0.35 +
        h_z * 0.30 +
        tr_z * 0.25 +
        (-dd_z) * 0.10
    )

    mean_rev = (
        (-ac_z) * 0.40 +
        (-h_z) * 0.30 +
        (-tr_z) * 0.20 +
        dd_z * 0.10
    )

    high_vol = (
        v_z * 0.70 +
        (-dd_z) * 0.20 +
        tr_z * 0.10
    )

    choppy = (
        (-trend_abs_z) * 0.40 +
        (-v_z) * 0.30 +
        (-autocorr_abs_z) * 0.30
    )

    raw = pd.DataFrame({
        "TRENDING": trending,
        "MEAN_REVERTING": mean_rev,
        "HIGH_VOLATILITY": high_vol,
        "CHOPPY": choppy,
    })

    # Clamp negatives to 0 before softmax-like normalization.
    # If a regime score is negative, that day simply doesn't look like
    # that regime at all — we don't want negative probabilities.
    raw_clamped = raw.clip(lower=0.0)

    # Normalize each row to sum to 1 (or to equal weight if all zeros)
    row_sums = raw_clamped.sum(axis=1)
    # If all four scores are 0 (no signal), assign equal probability (0.25 each)
    row_sums_safe = row_sums.replace(0.0, 1.0)
    probs = raw_clamped.div(row_sums_safe, axis=0)

    # Where the sum was truly 0, fill with equal weight
    all_zero = row_sums == 0.0
    probs.loc[all_zero] = 0.25

    return probs


def _feature_attribution(
    dominant_regime: str,
    feat_z_row: dict,
) -> list:
    """
    For the dominant regime on a given day, return the top-3 features by
    their absolute contribution to that regime's raw score, plus a human-
    readable reason string.
    """
    # Feature contribution = weight * z_score_value
    # We need the actual signed contribution for ranking
    regime_contrib = {
        "TRENDING": [
            ("autocorr_10", feat_z_row.get("autocorr_10_z", 0.0) * 0.35,
             "positive momentum autocorrelation"),
            ("hurst",       feat_z_row.get("hurst_z", 0.0) * 0.30,
             "Hurst > 0.5 — trending"),
            ("trend_20",    feat_z_row.get("trend_20_z", 0.0) * 0.25,
             "strong directional slope"),
            ("drawdown_20", (-feat_z_row.get("drawdown_20_z", 0.0)) * 0.10,
             "small drawdown consistent with uptrend"),
        ],
        "MEAN_REVERTING": [
            ("autocorr_10", (-feat_z_row.get("autocorr_10_z", 0.0)) * 0.40,
             "negative autocorrelation — prices reverse"),
            ("hurst",       (-feat_z_row.get("hurst_z", 0.0)) * 0.30,
             "Hurst < 0.5 — anti-persistent (mean-reverting)"),
            ("trend_20",    (-feat_z_row.get("trend_20_z", 0.0)) * 0.20,
             "flat slope — no directional bias"),
            ("drawdown_20", feat_z_row.get("drawdown_20_z", 0.0) * 0.10,
             "price stretched below recent peak"),
        ],
        "HIGH_VOLATILITY": [
            ("vol_20",      feat_z_row.get("vol_20_z", 0.0) * 0.70,
             "annualized volatility spike"),
            ("drawdown_20", (-feat_z_row.get("drawdown_20_z", 0.0)) * 0.20,
             "sharp drawdown accompanies vol spike"),
            ("trend_20",    feat_z_row.get("trend_20_z", 0.0) * 0.10,
             "directional move amid volatility"),
        ],
        "CHOPPY": [
            ("trend_20",    (-feat_z_row.get("trend_20_abs_z", 0.0)) * 0.40,
             "near-zero slope — no trend"),
            ("vol_20",      (-feat_z_row.get("vol_20_z", 0.0)) * 0.30,
             "low volatility — quiet market"),
            ("autocorr_10", (-feat_z_row.get("autocorr_10_abs_z", 0.0)) * 0.30,
             "near-zero autocorrelation — no serial dependence"),
        ],
    }

    contribs = regime_contrib.get(dominant_regime, [])
    # Sort by absolute contribution, descending
    contribs_sorted = sorted(contribs, key=lambda x: abs(x[1]), reverse=True)
    top3 = contribs_sorted[:3]

    result = []
    for feat_name, contrib_value, reason in top3:
        # The z-score for display: look up the base z-score
        z_key_map = {
            "autocorr_10": "autocorr_10_z",
            "hurst": "hurst_z",
            "trend_20": "trend_20_z",
            "vol_20": "vol_20_z",
            "drawdown_20": "drawdown_20_z",
        }
        z_key = z_key_map.get(feat_name, feat_name + "_z")
        z_val = feat_z_row.get(z_key, 0.0)
        result.append({
            "feature": feat_name,
            "z": round(float(z_val), 3),
            "reason": reason,
        })

    return result


def classify_regimes(prices: pd.DataFrame) -> dict:
    """
    Main public function: classify every trading day in `prices` into one of
    four regimes. Returns the full response payload (minus ticker/data_source,
    which main.py adds).

    Warm-up: the first 100 days are NaN-heavy (rolling z-scores need 100 days
    of history). These are dropped from the output so the frontend never
    sees nulls.

    Args:
        prices: DataFrame with a 'Close' column, indexed by date.

    Returns:
        dict with keys: dates, regimes, scores, features, prices,
        attribution_series, regime_counts, latest
    """
    if len(prices) < 100:
        raise ValueError(
            "Need at least 100 trading days for regime classification "
            f"(got {len(prices)})."
        )

    # --- Step 1: compute rolling features ------------------------------------
    feat = compute_features(prices)

    # --- Step 2: rolling z-score normalization (100-day past-only lookback) --
    # Each feature gets its own z-score series, so a "high vol" reading is
    # relative to the past 100 days, not the entire history.
    feat_z = pd.DataFrame(index=prices.index)
    for col in ["vol_20", "trend_20", "autocorr_10", "hurst", "drawdown_20"]:
        feat_z[col + "_z"] = _rolling_zscore(feat[col], window=100)

    # Additional derived z-scores for CHOPPY (absolute-value features)
    feat_z["trend_20_abs_z"] = _rolling_zscore(feat["trend_20"].abs(), window=100)
    feat_z["autocorr_10_abs_z"] = _rolling_zscore(feat["autocorr_10"].abs(), window=100)

    # --- Step 3: compute regime probabilities --------------------------------
    scores_df = _compute_regime_scores(feat_z)

    # --- Step 4: dominant regime per day -------------------------------------
    dominant = scores_df.idxmax(axis=1)

    # --- Step 5: drop warm-up days (first 100 rows) --------------------------
    # The z-score rolling window needs 100 days to produce meaningful results.
    # We also need 40 days for the Hurst exponent and 20 for trend/vol.
    # Dropping the first 100 rows covers all warm-up requirements.
    warmup = 100
    valid_idx = prices.index[warmup:]

    feat_valid = feat.loc[valid_idx]
    feat_z_valid = feat_z.loc[valid_idx]
    scores_valid = scores_df.loc[valid_idx]
    dominant_valid = dominant.loc[valid_idx]
    close_valid = prices["Close"].loc[valid_idx]

    # --- Step 6: build attribution series ------------------------------------
    attribution_series = []
    for i, date in enumerate(valid_idx):
        row_z = feat_z_valid.loc[date].to_dict()
        regime = dominant_valid.loc[date]
        attribution_series.append(_feature_attribution(regime, row_z))

    # --- Step 7: regime counts -----------------------------------------------
    regime_counts = {r: int((dominant_valid == r).sum()) for r in REGIMES}

    # --- Step 8: "latest" convenience block ----------------------------------
    last_date = valid_idx[-1]
    last_z_row = feat_z_valid.loc[last_date].to_dict()
    last_regime = dominant_valid.iloc[-1]
    latest = {
        "date": str(last_date.date()),
        "regime": last_regime,
        "scores": {r: round(float(scores_valid.loc[last_date, r]), 4) for r in REGIMES},
        "features": {
            col: round(float(feat_valid.loc[last_date, col]), 6)
            for col in ["vol_20", "trend_20", "autocorr_10", "hurst", "drawdown_20"]
        },
        "attribution": _feature_attribution(last_regime, last_z_row),
    }

    # --- Assemble final response dict ----------------------------------------
    return {
        "dates": [str(d.date()) for d in valid_idx],
        "regimes": list(dominant_valid),
        "scores": [
            {r: round(float(scores_valid.loc[d, r]), 4) for r in REGIMES}
            for d in valid_idx
        ],
        "features": {
            col: [round(float(v), 6) for v in feat_valid[col]]
            for col in ["vol_20", "trend_20", "autocorr_10", "hurst", "drawdown_20"]
        },
        "prices": [round(float(v), 4) for v in close_valid],
        "attribution_series": attribution_series,
        "regime_counts": regime_counts,
        "latest": latest,
    }


def regime_mask(prices: pd.DataFrame, regime_label: str) -> pd.Series:
    """
    Return a boolean Series (same index as prices) that is True on days
    classified as `regime_label`, False otherwise.

    IMPORTANT: the mask is already shift(1)ed — a regime detected on day T
    will be marked True on day T+1 (consistent with Bullpen's no-lookahead
    policy: you act on yesterday's signal).

    This is used by the /api/backtest regime_filter to AND strategy positions
    with the regime mask so that positions are only taken in the desired regime.

    Args:
        prices:       DataFrame with 'Close' column, indexed by date.
        regime_label: one of TRENDING, MEAN_REVERTING, HIGH_VOLATILITY, CHOPPY.

    Returns:
        Boolean Series, same index as prices, True where position is allowed.
    """
    if regime_label not in REGIMES:
        raise ValueError(
            f"Unknown regime '{regime_label}'. Must be one of: {REGIMES}"
        )

    # Get regime classifications (this covers the post-warmup period only)
    result = classify_regimes(prices)

    # Build a Series indexed by the full prices index, defaulting to False
    # (warm-up days before the first classified day default to False = no trade)
    mask = pd.Series(False, index=prices.index)

    # Map classified dates back to the mask
    classified_dates = pd.to_datetime(result["dates"])
    classified_labels = result["regimes"]

    for date, label in zip(classified_dates, classified_labels):
        if date in mask.index:
            mask.loc[date] = (label == regime_label)

    # Shift by 1 day: regime detected on day T → allowed to trade on day T+1.
    # This is the same no-lookahead convention as backtest.py's pos.shift(1).
    mask = mask.shift(1, fill_value=False)

    return mask
