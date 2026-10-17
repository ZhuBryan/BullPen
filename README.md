# Bullpen: Exchange Engine & Quantitative Research Platform

Two halves. A limit order book exchange built from scratch, and a research platform that uses it.

**The engine** ([`engine/`](engine/)) is a matching engine with correct price-time priority, written in Python and then ported to Rust. The two are differential-tested: fed identical order streams, they produce byte-identical event logs, and the Rust version runs **12.7x faster (3.4M orders/sec) with 13x lower p99.9 latency**.

Around it sits the rest of a trading stack: **FIX 4.4 order entry** and an OMS with cancel/replace and average-cost PnL, **pre-trade risk** (price collars, position limits, self-trade prevention), **instrument reference data** with Reg NMS tick regimes and multi-symbol routing, a **lock-free SPSC queue** for the feed-to-matcher handoff, **live Binance L2 capture** with sequence-gap detection and resync, **execution algorithms** (TWAP, VWAP, POV) scored by implementation shortfall, a **research pipeline** with purged walk-forward validation, and **latency histograms** exposed as Prometheus text.

**The platform** is a paper-trading arena and backtester: write trading rules in plain English or a small safe DSL, race strategies against each other, and stress-test them with a parameter optimizer, walk-forward validation, Monte Carlo, and a market-regime detector.

No paid APIs, no API keys. Free Yahoo Finance data with a synthetic offline fallback.

**Live demo: [bullpen-858e.onrender.com](https://bullpen-858e.onrender.com)**

There is a [technical writeup](WRITEUP.md) covering the optimization work and, more usefully, the parts I got wrong.

## Quick start

The engine and its tests (standard library plus numpy):

```bash
cd engine
python test.py          # every self-check
python bench.py         # throughput and tail latency
cd rust && cargo test   # the Rust port
```

The web platform:

```bash
pip install -r requirements.txt
python run.py
```

Then open http://localhost:8000.

## Hosting

The live demo runs on [Render](https://render.com)'s free tier. Free instances sleep after about 15 minutes idle and take 30 to 50 seconds to wake, so a GitHub Actions job (`.github/workflows/keep-warm.yml`) pings the site every 10 minutes to keep it awake for visitors.

To host your own copy: fork the repo, create a Render account, choose New + then Blueprint, and connect the fork. Render reads `render.yaml` for the build and start commands. Then set an `APP_URL` repository variable to your new URL so the keep-warm job pings yours instead of mine.

## Arena tab: watch strategies fight it out

Five bots (Warren the trend-follower, Dipsy the dip-buyer, Momo the momentum chaser, HODL the buy-and-holder, and Flip the fast SMA-flipper) race each other on a real ticker. History replays at about two seconds per trading day, and the equity chart, price candlesticks, and leaderboard all update live.

You can add your own fighter too. Write a buy rule and a sell rule using a small, safe expression language. There's no `eval` anywhere; rules go through a whitelisted parser in `backend/rules.py`. A few examples:

- Dip buyer: buy `RSI(14) < 30 AND CLOSE > SMA(200)`, sell `RSI(14) > 70`
- Trend follower: buy `SMA(20) > SMA(50)`, sell `SMA(20) < SMA(50)`
- Buy and hold: buy `CLOSE > 0`, sell `CLOSE < 0`

The indicators you can use are `CLOSE`, `SMA(n)`, `EMA(n)`, `RSI(n)`, and `MOM(n)`, combined with `AND`, `OR`, and the comparisons `< > <= >=`. The in-app cheat sheet has the full grammar and click-to-fill examples.

## Lab tab: backtest a strategy on history

Pick a ticker and date range, choose a built-in strategy (Buy and Hold, SMA Crossover, RSI Mean Reversion, Bollinger Bands, or Momentum) or write custom rules with the same language as the Arena. You get a candlestick price chart with your buy and sell points marked, an equity curve against buy and hold, a drawdown chart, a monthly returns heatmap, and the key metrics below.

## Analysis tab: go beyond a single backtest

Three tools that come up in real quant interviews:

**Parameter optimizer.** Instead of guessing whether SMA(10, 50) or SMA(20, 100) is better, grid-search a range of values and read off a heatmap of which combination maximized Sharpe ratio (or CAGR, Calmar, and so on). The UI is upfront about the catch: a parameter that looks great on this data may just be overfit to it.

**Walk-forward validation.** This is the honest way to check whether an optimization was real. Split the date range into folds, tune parameters on each training window, then test on the next window the tuner never saw. The equity curve it shows is purely out of sample.

**Monte Carlo simulation.** Reshuffle the strategy's actual daily returns thousands of times to show the range of outcomes it could have produced. It answers whether a track record is repeatable or just lucky, with percentile bands and a plain-English probability of loss. It isn't a forecast of the future. It measures how much the result depended on the exact order of good and bad days.

## Regime tab: market regime detection with explanations

This is the part I'm most proud of, because it goes past a standard backtest.

It classifies each trading day into one of four market regimes (Trending, Mean-Reverting, High-Volatility, or Choppy) by computing five rolling statistical features from price data:

| Feature | What it measures |
|---|---|
| `vol_20` | 20-day annualized volatility, or how wildly prices are swinging |
| `trend_20` | Directional slope of prices over 20 days |
| `autocorr_10` | Lag-1 autocorrelation: do today's gains predict tomorrow's, or reverse them? |
| `hurst` | Hurst exponent: above 0.5 trends, below 0.5 mean-reverts, near 0.5 is a random walk |
| `drawdown_20` | Current distance below the recent 20-day peak |

There's no `.fit()` call here and no learned weights. Classification uses a fixed affinity matrix built from domain knowledge, applied to rolling z-scores. Every regime call comes with the three features that drove it and a reason in plain English, for example "autocorr_10 = +0.42 (positive momentum autocorrelation)". The explainability is the point; anyone can call a black-box classifier.

The regime-gated backtest ties the detector into the rest of Bullpen. Run any built-in strategy, but only take positions when the detector says the market is in a chosen regime. The UI reports what fraction of days the strategy was active and whether filtering actually improved risk-adjusted returns.

A few limitations worth being honest about:

- Regime labels are heuristic, not ground truth. There's no universal definition of "trending," so treat the labels as input to your judgment, not a verdict.
- Features are computed on the same asset being traded, so at daily resolution there's a small lookahead risk. The regime for day T uses day T's close, and Bullpen only acts on it from day T+1.
- Regime-gating trades on fewer days, which can shrink drawdowns just by trading less. Compare against unfiltered buy and hold, not only the unfiltered strategy.

## Compare tab: line up several tickers

Run one strategy across up to six tickers at once on a shared date range. You get all their equity curves on one chart, a summary metrics table, and a return-correlation heatmap that shows how closely each pair moves together day to day. Low correlations are the ones that actually diversify risk.

## engine/: the matching engine underneath

Everything above treats prices as something that already happened. `engine/` is the other half: a limit order book exchange built from scratch, where prices are produced rather than downloaded.

- **Matching engine** with real price-time priority, integer tick prices, partial fills, and lazy cancellation. 269,000 orders/sec single-threaded in pure Python, with a p99 of 12.6 microseconds.
- **Live capture** of Binance L2 depth and trades over the public websocket, including the snapshot/diff handshake and sequence-gap detection that keep a local book honest.
- **Replay with queue position**, so a backtest asks "would my order actually have filled, given the 400 shares queued ahead of it" instead of assuming a fill at mid.
- **Purged walk-forward research**, which drops training rows whose labels overlap the test window. Ordinary k-fold leaks here and inflates every metric.

```bash
cd engine
python test.py     # all self-checks
python bench.py    # throughput and tail latency
```

See `engine/README.md` for the benchmark methodology and the before/after numbers, including the profiling mistake that led to the 28x tail-latency fix.

## The metrics, in plain English

- **Total return:** how much you made versus just holding your starting cash.
- **Buy and hold return:** what doing nothing clever would have earned. If you can't beat this, the strategy isn't worth it.
- **CAGR:** the same return expressed as a steady percent per year.
- **Sharpe ratio:** return per unit of risk. Rough guide: above 1 is good, above 2 is great, above 3 and you should check for a bug.
- **Sortino ratio:** like Sharpe, but it only penalizes downside swings, not the upside ones you actually want.
- **Max drawdown:** the worst peak-to-bottom loss along the way. The "could you stomach this?" number.
- **Calmar ratio:** annual return divided by that worst drawdown.
- **Win rate:** the percent of trades that made money.

## One detail that makes this legit: no lookahead bias

A signal computed from today's close can only be traded tomorrow. The engine shifts every position forward one day (see `backtest.py`). Backtests that skip this step look amazing and are completely fake, so getting it right matters more than it seems.

## What's inside

| File | What it does |
|---|---|
| `backend/data.py` | Downloads prices for free, caches them, and falls back to synthetic data offline |
| `backend/strategies.py` | Built-in strategies: Buy and Hold, SMA Crossover, RSI Mean Reversion, Bollinger Bands, Momentum |
| `backend/rules.py` | Safe parser for user-written buy and sell rules, with no `eval` or `exec` |
| `backend/backtest.py` | The engine: simulates trading and computes Sharpe, Sortino, Calmar, drawdown, and win rate |
| `backend/analysis.py` | Parameter optimizer, walk-forward validation, and Monte Carlo simulation |
| `backend/regime.py` | Market regime detector: rolling features, affinity-matrix classifier, feature attributions |
| `backend/arena.py` | The live arena: replay clock, default fighters, leaderboard, agent persistence |
| `backend/main.py` | FastAPI server and all JSON API endpoints |
| `frontend/index.html` | The web UI: Arena, Lab, Analysis, Regime, and Compare tabs, built with Chart.js and vanilla JS |
| `backend/ai.py` | Turns a plain-English strategy description into validated rules, using your own API key or a local model |
| `backend/test_bullpen.py` | Plain-Python test suite, run with `python backend/test_bullpen.py` |
| `engine/` | The matching engine, live capture, queue-position replay, and research pipeline |

## Roadmap

1. ~~Backtesting engine and web UI~~ (done)
2. ~~Custom strategy rule language~~ (done)
3. ~~Live arena with leaderboard~~ (done)
4. ~~Free deploy on the Render free tier~~ (done)
5. ~~Parameter optimizer, walk-forward validation, and Monte Carlo simulation~~ (done)
6. ~~Market regime detector with explainability and regime-gated backtesting~~ (done)
7. ~~Candlestick charts, volume, and a multi-ticker correlation heatmap~~ (done)
