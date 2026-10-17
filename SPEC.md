# Bullpen SPEC — Arena + Custom Strategy DSL

Contract between backend and frontend. Both sides build against THIS file.

## 1. Strategy rule language (DSL)

Users write two rules as plain text. NO eval/exec — whitelisted parser only
(`backend/rules.py`).

- **buy_rule**: when flat and this is true → go long next day
- **sell_rule**: when long and this is true → go to cash next day

Grammar (case-insensitive):

```
expr       := orexpr
orexpr     := andexpr ( OR andexpr )*
andexpr    := comparison ( AND comparison )*
comparison := term ( < | > | <= | >= ) term
term       := NUMBER | indicator | '(' expr ')'   # parens allowed around exprs
indicator  := CLOSE | SMA(n) | EMA(n) | RSI(n) | MOM(n)
```

Indicators (computed on daily closes):
- `CLOSE` — today's closing price
- `SMA(n)` / `EMA(n)` — simple / exponential moving average, n days
- `RSI(n)` — relative strength index 0–100
- `MOM(n)` — % return over last n days (e.g. `MOM(60) > 0` = up over 60 days)

Examples:
- buy `RSI(14) < 30 AND CLOSE > SMA(200)`, sell `RSI(14) > 70`
- buy `SMA(20) > SMA(50)`, sell `SMA(20) < SMA(50)`
- buy&hold: buy `CLOSE > 0`, sell `CLOSE < 0`

Parse errors must return a human-readable message pointing at the problem
token. AND binds tighter than OR. n must be an integer 1–500. While an
indicator is still warming up (NaN), the comparison is false.

## 2. Arena mechanics (`backend/arena.py`)

- One global arena in server memory. Config: ticker (default "SPY"),
  ~2 years of daily data via `data.get_prices`, `seconds_per_day = 2.0`.
- Replay clock: `day_index = min(start_offset + elapsed_seconds / seconds_per_day, len(data)-1)`,
  computed lazily whenever state is requested (no background threads).
  `start_offset = 60` so indicators are warmed up at day 0 of the race.
- Every agent (default + user-created) is a pair of DSL rules evaluated by
  the same engine. Positions: signal from day T trades at day T+1's return
  (no lookahead). Commission 0.1% per position change. Start cash $10,000.
- Equity series are precomputed for the full window at agent creation /
  reset (cheap), then *revealed* up to day_index — so state requests are fast.
- When day_index hits the end: season ends; auto-restart (re-fetch data,
  same agents, equities reset).
- 5 default personalities (defined as DSL rules, with emoji + tagline),
  e.g. Warren 🧓 trend follower, Dipsy 🤿 dip buyer, Momo 🚀 momentum,
  HODL 💎 never sells, Flip 🪙 fast SMA flipper. Defaults are not deletable.
- Custom agents persist in `data/agents.json` (id, name, emoji, rules, created_at).

## 3. API endpoints (added to `backend/main.py`)

### POST /api/rules/validate
Body: `{"buy_rule": str, "sell_rule": str}` →
`{"valid": true}` or `{"valid": false, "error": "msg"}` (HTTP 200 either way).

### GET /api/arena/state
```json
{
  "ticker": "SPY", "data_source": "yahoo|cache|sample",
  "day_index": 123, "total_days": 440, "season": 1,
  "date": "2025-03-04",
  "dates": ["..."],            // dates[0..day_index] only
  "agents": [{
    "id": "warren", "name": "Warren", "emoji": "🧓",
    "tagline": "The trend is your friend",
    "is_default": true,
    "buy_rule": "SMA(20) > SMA(50)", "sell_rule": "SMA(20) < SMA(50)",
    "equity_series": [10000.0, ...],   // aligned with dates
    "equity": 11234.56, "return_pct": 12.35,
    "position": "LONG" | "CASH", "rank": 1
  }]
}
```
Agents sorted by rank (1 = highest equity).

### POST /api/arena/agents
Body: `{"name": str (<=24 chars), "emoji": str (optional), "buy_rule": str, "sell_rule": str}`
→ 200 with the new agent object, or 400 `{"detail": "msg"}` on bad rules/name.
New agent's equity starts at $10,000 *from the current day* (flat $10,000
series before its join day).

### DELETE /api/arena/agents/{id}
Custom agents only; defaults → 400. Returns `{"ok": true}`.

### POST /api/arena/reset
Body: `{"ticker": str (optional)}` — new season, optionally new ticker.
Returns same shape as /api/arena/state.

### POST /api/backtest (EXTENDED, Lab tab)
Now also accepts `"strategy": "custom"` with
`"params": {"buy_rule": str, "sell_rule": str}` → same response as before.
Existing built-in strategies unchanged.

## 4. Frontend (`frontend/index.html`, single file)

Three tabs, same dark theme:
- **Arena** (default): live leaderboard (rank, emoji, name, tagline, return %,
  equity, LONG/CASH badge, delete button on custom agents), equity race chart
  (one line per agent, poll /api/arena/state every 2s), "Day X / Y · season N"
  header, ticker + reset control, and a "New fighter" form: name, emoji,
  buy/sell rule inputs with quick-insert chips (RSI(14), SMA(50), …),
  live validation via /api/rules/validate, cheat-sheet of the grammar.
- **Lab**: the existing backtester, plus "Custom rules" in the strategy
  dropdown showing the two rule inputs.
- **Analysis** (new): three sub-views over the same ticker/date/strategy
  inputs as Lab — Optimizer (heatmap/bar chart of a param grid sweep, see
  POST /api/optimize), Walk-Forward (out-of-sample equity curve + per-fold
  table, see POST /api/walkforward), and Monte Carlo (fan chart of
  percentile equity bands + risk stats, see POST /api/montecarlo). These
  are "serious quant" views — favor clear labeling of what's in-sample vs
  out-of-sample, and plain-English explanations of each stat (a beginner
  should understand why walk-forward / Monte Carlo matter).

## 5. Analysis endpoints (`backend/main.py`, new module e.g. `backend/analysis.py`)

All three below operate on **built-in strategies only** (not `"custom"` DSL)
for Optimizer and Walk-Forward, since they need a numeric `param_grid` to
sweep — `STRATEGIES[id]["params"]` already declares each strategy's tunable
numeric params (name + default). Monte Carlo works for ANY strategy
(built-in or `"custom"`) since it only needs the resulting daily-return
series from one ordinary backtest.

Shared safety limits (return HTTP 400 with a clear message if exceeded):
- `param_grid`: at most 2 params, each with at most 12 values → grid size
  <= 144 combinations.
- Optimizer `objective` / Walk-Forward `objective`: one of
  `"sharpe" | "sortino" | "calmar_ratio" | "cagr_pct" | "total_return_pct"`
  (default `"sharpe"`), higher = better for all of these.
- `n_sims` (Monte Carlo): 100–2000, default 500.
- `n_folds` (Walk-Forward): 2–8, default 4.

### POST /api/optimize
Body:
```json
{
  "ticker": "AAPL", "start": "2020-01-01", "end": "2025-01-01",
  "strategy": "sma_crossover",
  "param_grid": {"fast": [5,10,15,20], "slow": [50,100,150,200]},
  "objective": "sharpe",
  "starting_cash": 10000.0
}
```
Response:
```json
{
  "strategy": "SMA Crossover", "objective": "sharpe",
  "param_names": ["fast","slow"],
  "results": [
    {"params": {"fast":5,"slow":50}, "metrics": {...full metrics dict from run_backtest...}},
    ...
  ],
  "best": {"params": {"fast":10,"slow":100}, "metrics": {...}}
}
```
`results` covers the full cartesian product of `param_grid` (in order).
Frontend renders a heatmap if 2 params, a bar chart if 1 param.

### POST /api/walkforward
Body: same shape as `/api/optimize` plus `"n_folds": 4`.

Method (rolling walk-forward optimization): split the full date range into
`n_folds + 1` equal consecutive segments `S0..Sn`. For fold `i` in
`0..n_folds-1`: grid-search `param_grid` on train segment `Si` (pick the
combo maximizing `objective`), then run a backtest with those params on the
*next* segment `S(i+1)` — this is out-of-sample. Stitch the out-of-sample
equity curves end-to-end (each fold's curve starts where the previous one's
ended, i.e. compound the daily returns continuously) to get one combined
out-of-sample equity curve covering segments `S1..Sn`.

Response:
```json
{
  "strategy": "SMA Crossover", "objective": "sharpe", "n_folds": 4,
  "folds": [
    {
      "train_range": ["2020-01-01","2021-01-06"],
      "test_range": ["2021-01-06","2022-01-11"],
      "best_params": {"fast":10,"slow":100},
      "train_metrics": {...},   // metrics of best_params ON the train segment
      "test_metrics": {...}     // out-of-sample metrics ON the test segment
    }, ...
  ],
  "combined": {
    "dates": [...], "equity": [...], "buy_hold": [...],
    "drawdown_pct": [...], "metrics": {...}   // same shape as run_backtest output
  }
}
```
If a segment is too short for the strategy's longest lookback (e.g. SMA(200)
needs >=200 days), return HTTP 400 with a message suggesting fewer folds or
a longer date range.

### POST /api/montecarlo
Body:
```json
{
  "ticker": "AAPL", "start": "2020-01-01", "end": "2025-01-01",
  "strategy": "sma_crossover", "params": {"fast":10,"slow":50},
  "starting_cash": 10000.0, "n_sims": 500
}
```
(`strategy: "custom"` with `params: {"buy_rule","sell_rule"}` also allowed,
same as `/api/backtest`.)

Method: run one normal backtest to get the strategy's daily-return series
(post-shift, post-commission). Then `n_sims` times, resample those daily
returns WITH REPLACEMENT (bootstrap) into a new sequence of the same length,
compound into a simulated equity curve, and record final equity / total
return % / max drawdown % for that simulation. This answers "if the days
that actually happened had occurred in a different order / with different
luck, how would this strategy's results have varied?" — NOT a prediction of
the future, and the frontend must say so.

Response:
```json
{
  "strategy": "SMA Crossover", "n_sims": 500, "starting_cash": 10000.0,
  "original": {"final_equity":12345.67,"total_return_pct":23.46,"max_drawdown_pct":-18.2},
  "percentiles": {
    "p5":  {"final_equity":..., "total_return_pct":..., "max_drawdown_pct":...},
    "p25": {...}, "p50": {...}, "p75": {...}, "p95": {...}
  },
  "bands": {
    "day": [0,1,2,...],          // trading-day index (0-based, not calendar dates -- order is randomized)
    "p5": [...], "p25": [...], "p50": [...], "p75": [...], "p95": [...]
  },
  "prob_loss_pct": 12.4,           // % of sims ending below starting_cash
  "prob_beat_buy_hold_pct": 64.8   // % of sims whose total_return_pct beat the ORIGINAL buy&hold return
}

## 6. Regime detector (`backend/regime.py`, new module)

### What it does
Classifies each trading day into one of four market regimes based on rolling
statistical features of price behavior — with no ML model that could "learn"
the future. Every classification is fully explainable: each day comes with the
feature values that drove the call.

### Four regimes (string labels used in all API responses)
| Label | Meaning |
|---|---|
| `TRENDING` | Prices have a clear directional bias; momentum strategies tend to work |
| `MEAN_REVERTING` | Prices oscillate around a level; dip-buying tends to work |
| `HIGH_VOLATILITY` | Unusually large daily swings regardless of direction |
| `CHOPPY` | No clear trend, no reversion signal, low predictability |

### Rolling features (computed on daily Close prices, NO lookahead)
All windows look only at data up to and including day T; regimes are known on
day T but only *acted on* at day T+1 in backtests (consistent lookahead policy
with the rest of Bullpen).

| Feature key | Formula / description | Typical range |
|---|---|---|
| `vol_20` | 20-day rolling σ of daily log-returns, annualized × √252 | 0.05 – 1.0 |
| `trend_20` | Absolute slope of OLS fit to log-prices over 20 days, scaled to daily % | 0 – 0.05 |
| `autocorr_10` | Pearson autocorrelation of daily returns at lag-1, 10-day window | -1 – 1 |
| `hurst` | Hurst exponent estimate via variance-time method, 40-day window. <0.5 = mean-reverting, ~0.5 = random walk, >0.5 = trending | 0 – 1 |
| `drawdown_20` | Current drawdown from the rolling 20-day price peak, as a fraction | -1 – 0 |

Warm-up: the first `max(40, …)` days will have NaN features and should be
returned as regime `null` (or omitted from the dates list — pick whichever
makes the frontend simpler). The feature that is still NaN never contributes
to that day's regime score.

### Classification algorithm (pure numpy/pandas, no sklearn)
1. For each feature, compute a **rolling z-score** vs. the prior 100 trading
   days (using only past data: `rolling(100).mean()` and `.std()`, not the full
   sample). This normalizes each feature to ~N(0,1) without lookahead.
2. Apply a fixed **affinity matrix** (domain knowledge, not learned) mapping
   each (feature, regime) pair to a weight: e.g., high `autocorr_10` → strong
   `TRENDING` score; negative `autocorr_10` → `MEAN_REVERTING`; high `vol_20`
   → `HIGH_VOLATILITY`; all features near 0 → `CHOPPY`. The exact weights are
   your engineering judgment — document them as constants in `regime.py`.
3. Clamp raw scores to [0, ∞) and normalize across regimes so they sum to 1
   (soft probabilities). The dominant regime is `argmax`.
4. For **feature attribution**: for the dominant regime, sort the features by
   their contribution to that regime's raw score, descending. Return the top 3
   as `attribution` with a `reason` string (e.g., `"autocorr_10 = +0.42 (positive momentum autocorrelation)"`).

### POST /api/regime
Body:
```json
{ "ticker": "SPY", "start": "2020-01-01", "end": "2025-01-01" }
```
Response:
```json
{
  "ticker": "SPY", "data_source": "yahoo|cache|sample",
  "dates": ["2020-04-15", ...],           // warm-up days excluded
  "regimes": ["TRENDING", "CHOPPY", ...], // one per date
  "scores": [
    {"TRENDING": 0.62, "MEAN_REVERTING": 0.08, "HIGH_VOLATILITY": 0.21, "CHOPPY": 0.09},
    ...
  ],
  "features": {
    "vol_20":      [0.18, 0.20, ...],
    "trend_20":    [0.012, 0.015, ...],
    "autocorr_10": [0.31, 0.28, ...],
    "hurst":       [0.57, 0.58, ...],
    "drawdown_20": [-0.01, -0.03, ...]
  },
  "prices": [312.5, 314.1, ...],          // close prices on those same dates (for overlay chart)
  "attribution_series": [                 // one entry per date, top-3 feature drivers for that day's call
    [
      {"feature": "autocorr_10", "z": 1.4, "reason": "positive momentum autocorrelation"},
      {"feature": "hurst",       "z": 0.9, "reason": "Hurst > 0.5 — trending"},
      {"feature": "trend_20",    "z": 0.7, "reason": "strong directional slope"}
    ],
    ...
  ],
  "regime_counts": {"TRENDING": 210, "MEAN_REVERTING": 89, "HIGH_VOLATILITY": 54, "CHOPPY": 147},
  "latest": {                             // most recent date's data (convenience for the UI)
    "date": "2024-12-31", "regime": "TRENDING",
    "scores": {...}, "features": {...},
    "attribution": [...]
  }
}
```
HTTP 400 if fewer than 150 trading days of data (not enough for warm-up).

### POST /api/backtest — regime_filter extension
Add one optional field to the existing `BacktestRequest` model:
```
"regime_filter": null | "TRENDING" | "MEAN_REVERTING" | "HIGH_VOLATILITY" | "CHOPPY"
```
When non-null: compute regimes for the same ticker/date range, then AND the
strategy's positions with a daily mask `(regime == regime_filter)`. Days outside
the filter regime are forced to cash (position = 0). The response gains two
extra fields alongside the normal backtest output:
```json
{
  ...normal backtest fields...,
  "regime_filter": "TRENDING",
  "regime_fraction_pct": 38.2   // % of days in that regime (so user knows how often strategy was active)
}
```
The unfiltered buy & hold baseline (`buy_hold`) is unchanged (not regime-gated),
so the user can see whether the regime-gating beat passive holding.
```

## 7. Roadmap phase (CI, Rust in the product, book backtests, research)

Built in parallel by separate agents. **Each agent edits only the files it
owns.** Shared files (`backend/main.py`, `frontend/index.html`,
`engine/test.py`, `backend/test_bullpen.py`, `requirements.txt`,
`render.yaml`, `.github/`, `README.md`, `WRITEUP.md`, `NOTES.md`) are owned
by the integrator, who wires endpoints, registers tests and commits. Agents
never commit.

Conventions for every module (from CLAUDE.md): stdlib/numpy/pandas only (no
new dependencies except where listed below), no paid APIs, beginner-friendly
comments that explain the finance, signals on day T trade on day T+1, and one
`demo()` self-check with `assert`s that `python -m <module>` runs and that
raises on failure. Prices inside the engine are integer ticks.

### 7.1 Portable book interface

Both `engine.orderbook.Book` (Python) and the PyO3 `bullpen_engine.Book`
(Rust) expose exactly this. Simulator and backtester code use nothing else.

| Method | Returns |
|---|---|
| `limit(side, price, qty)` | order id (int) |
| `market(side, qty)` | order id (int) |
| `cancel(order_id)` | bool |
| `best_bid()` / `best_ask()` | int ticks or `None` |
| `depth(side, levels=5)` | list of `(price, live_qty)`, best first |
| `event_count()` | int |
| `events_since(n)` | list of event tuples from index n |
| `has_order(order_id)` | bool (resting and live) |
| `order_ids()` | list of live resting ids |

`side` is `"B"` or `"S"`. Event tuples are identical to the Python engine's,
including trailing `seq` and `ts` (ts may differ between engines; seq may not).
`engine/fastbook.py` exports `Book` (Rust when the extension imports, else
Python) and `ENGINE = "rust" | "python"`.

### 7.2 Ownership

| Agent | Owns (creates or edits) | Delivers |
|---|---|---|
| A, Rust | `engine/rust/**`, `engine/fastbook.py`, `engine/test_pyo3.py` | PyO3 binding behind a cargo feature (core crate stays dependency-free), maturin `pyproject.toml`, criterion benches in a separate crate, differential test Python-vs-binding |
| B, Microstructure | `backend/market_sim.py`, `backend/stylized.py` | Avellaneda-Stoikov maker with PnL decomposition; stylized-facts analysis of real vs simulated returns, plus a capture-file path |
| C, Book backtest | `backend/book_backtest.py` | Event-driven backtester whose orders fill against a modelled order book, with per-trade TCA |
| D, Portfolio | `backend/portfolio.py` | Multi-asset backtest, position sizing, VaR/CVaR, beta, factor exposure |
| E, Infra | `engine/fix_session.py`, `engine/bench.py` | FIX 4.4 session layer over TCP; benchmark rigour (repeats, spread, JSON) |

### 7.3 Endpoints the integrator will add

Each agent's module exposes one top-level function returning a JSON-ready
dict; the integrator wraps it.

- `POST /api/backtest` gains `fill_model: "close" | "book"` -> `book_backtest.run(...)`
- `POST /api/sim/maker` -> `market_sim.run_maker(...)`
- `POST /api/stylized` -> `stylized.compare(...)`
- `POST /api/portfolio` -> `portfolio.run(...)`
- `GET  /api/metrics` gains `bullpen_engine_info{engine="rust|python"} 1`
