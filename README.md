# Bullpen: Exchange Engine & Quantitative Research Platform

[![CI](https://github.com/ZhuBryan/BullPen/actions/workflows/ci.yml/badge.svg)](https://github.com/ZhuBryan/BullPen/actions/workflows/ci.yml)

Two halves. A limit order book exchange built from scratch, and a research platform that uses it.

**The engine** ([`engine/`](engine/)) is a matching engine with correct price-time priority, written in Python and then ported to **Rust and C++17**. All three are differential-tested: fed identical order streams, they produce byte-identical event logs. The Rust version runs **12.7x faster than Python (3.4M orders/sec) with 13x lower p99.9 latency**; the C++ port is clean under AddressSanitizer and UBSan and builds with both g++ and MSVC in CI.

Around it sits the rest of a trading stack: **FIX 4.4 over TCP** with a real session layer (heartbeats, sequence-gap recovery, resend and gap fill) feeding an OMS with cancel/replace and average-cost PnL, **pre-trade risk** (price collars, position limits, self-trade prevention), **instrument reference data** with Reg NMS tick regimes and multi-symbol routing, a **lock-free SPSC queue** for the feed-to-matcher handoff, **live Binance L2 capture** with sequence-gap detection and resync, **execution algorithms** (TWAP, VWAP, POV) scored by implementation shortfall, an **Avellaneda-Stoikov market maker**, and **latency histograms** exposed as Prometheus text.

The Rust engine isn't only a benchmark: it's exposed to Python through **PyO3**, and the app's simulator runs on it whenever the extension is built. End to end, a 2,000-step market simulation runs **about 1.5x faster** on the Rust book (median 0.117s vs 0.077s, best 0.110s vs 0.067s; identical 3,639 trades). That's far below the engine's 12.7x, but close to the ~1.8x ceiling for this simulator, and measuring says why. `python -m engine.bench_sim --profile` records every call the simulation makes on the book, replays them in a bare loop, and separately times the Python side against a no-work stand-in: the book is about 48% of a run, so by Amdahl's law even an infinitely fast engine caps the gain near 1.9x. Orders and cancels now cross into Rust in batches, which trims per-call overhead, but raising the ceiling means moving the agents' decision logic out of Python. **CI** builds it on every push, proves it matches the Python engine event for event, and then reruns the whole backend test suite on top of it.

**The platform** is a paper-trading arena and research workbench: write trading rules in plain English or a small safe DSL, race strategies against each other, and stress-test them with a parameter optimizer, walk-forward validation, Monte Carlo, and a market-regime detector. Backtests can fill against a modelled **order book** instead of the closing price, multi-asset **portfolios** get proper risk numbers (VaR, CVaR, beta, factor exposure), and a **stylized-facts** check asks whether the simulated market behaves like a real one.

No paid APIs, no API keys. Free Yahoo Finance data with a synthetic offline fallback.

**Live demo: [bullpen-858e.onrender.com](https://bullpen-858e.onrender.com)**

There is a [technical writeup](https://zhubryan.github.io/BullPen/WRITEUP) covering the optimization work and, more usefully, the parts I got wrong.

## Quick start

The engine and its tests (standard library plus numpy):

```bash
cd engine
python test.py          # every self-check
python bench.py         # throughput and tail latency, 5 repeats with spread
cd rust
cargo test              # the Rust port
cd ../cpp
make test               # the C++ port (or see engine/cpp/README.md for MSVC)
```

To record real order-book data: `python capture.py --live kraken BTC/USD 3600` from `engine/`, then `python research.py "data\kraken_BTCUSD_*.jsonl.gz"` to run the order-flow forecast on it.

To run the app on the Rust engine, build the Python extension once: inside `engine/rust`, run `python -m pip install .` (pip fetches maturin and compiles the Rust for you). Without it everything still works on the Python engine.

To run the tests yourself, install the test dependencies first: `pip install -r requirements-dev.txt`.

On Windows PowerShell, run these as separate lines; older PowerShell doesn't accept `&&`.

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

**Fills: close or order book.** The default fills every trade at the close, which is the usual idealization. Switch to *Order book* and each signal is sent as a market order at the next day's open into a book built from that day's spread and trailing volume, so a big order walks the book and pays impact. You get per-trade transaction cost analysis and the gap versus close fills split into four parts that add up exactly:

| Part | What it isolates |
|---|---|
| Timing | Trading at the next open instead of the signal day's close, with no frictions at all |
| Rounding | Buying whole shares instead of fractions |
| Cost | Spread and impact actually paid |
| Capacity | Size the book couldn't absorb (or "other" when every order filled) |

The split matters because the raw gap can mislead. On a losing strategy, an account too big to get filled can look like it did *better* than close fills; here that shows up as capacity, not as cheap execution.

One caveat applies to every backtest here: the tickers you pick are today's survivors. Companies that went bust aren't in the list, which flatters any strategy that buys and holds.

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

## Portfolio tab: several assets at once

Pick 2 to 10 tickers and a sizing scheme:

- **Equal weight**: the naive baseline that's famously hard to beat.
- **Inverse volatility**: calmer assets get more money, so each moves the account by a similar amount.
- **Volatility target**: inverse-vol weights scaled so the forecast portfolio volatility hits a target. It never borrows, so in calm markets it just holds cash.
- **Risk parity**: every asset contributes the same share of total risk, solved exactly (including when assets are negatively correlated, which a common shortcut gets wrong).

On real data (SPY, TLT, GLD and QQQ, 2015 to 2024) every scheme landed at a Sharpe ratio of about 0.92: risk-based sizing cut volatility and drawdown (vol targeting took max drawdown from -25.6% to -20.4%) but didn't improve return per unit of risk, and monthly rebalancing gave up about 38 points of return against buy and hold by repeatedly trimming QQQ. That's the honest, textbook result, and the tool shows it rather than hiding it.

Weights are decided with data up to day T and held from T+1, rebalanced monthly or weekly. Results include the equity curve against equal-weight buy and hold, weights over time, historical and parametric VaR and CVaR at 95% and 99%, beta to SPY, a factor regression on free ETF proxies (market, size, value, momentum) with t-stats, the correlation matrix, and each asset's share of portfolio risk.

## Engine tab: a market that produces its own prices

- **Simulated market.** Agents (market makers, momentum and mean-reversion traders, noise traders) post orders into the matching engine and trade with each other. Closed-economy invariants are checked every step: shares and cash are conserved, the book never crosses.
- **Execution and TCA.** Work a parent order with TWAP, VWAP, or POV against that live book and score it by implementation shortfall.
- **Market making.** An Avellaneda-Stoikov maker, which skews its quotes against its inventory, runs head to head with a naive symmetric one on the same market. Profit splits exactly into spread capture and inventory PnL. A-S holds about 3x less inventory and runs near break-even, while the naive maker earns more by carrying more risk: reducing risk has a price.
- **Stylized facts.** Real returns and the simulator's returns are tested against five well-documented properties of markets: heavy tails, no linear autocorrelation, volatility clustering, aggregational Gaussianity, and the leverage effect. The simulator used to fail the autocorrelation test badly (lag-1 correlation of +0.35). Ablations traced it to market makers anchoring their quotes to their own previous quotes, which turned every inventory adjustment into a trend; after fixing that at the root, it passes on 87% of held-out seeds. It still has no leverage effect, and says so.
- **Order-flow forecast.** A short-horizon forecast of the mid price from order-flow imbalance, book imbalance, the microprice and recent trade flow, refitted walk-forward with a purge gap and scored only out of sample, next to a negative control fed an unrelated market's features. It has real skill (information coefficient around 0.2 a few steps ahead, control near zero). A market maker that leans its quotes on the forecast is compared with plain A-S on the same market using markouts, the standard measure of adverse selection; so far the lean doesn't reliably pay, and the card reports that as it is.
- **Research agent.** Searches strategy variants and then discounts its own best result with the Deflated Sharpe Ratio, because search enough variations and one will look brilliant by luck.

## engine/: the matching engine underneath

Everything above treats prices as something that already happened. `engine/` is the other half: a limit order book exchange built from scratch, where prices are produced rather than downloaded.

- **Matching engine** with real price-time priority, integer tick prices, partial fills, and lazy cancellation. 269,000 orders/sec single-threaded in pure Python, with a p99 of 12.6 microseconds.
- **Live capture** of L2 depth and trades from **Kraken** and Binance. The two keep a local book honest in opposite ways: Binance numbers every update, so you detect a *missed message* before applying anything; Kraken sends a CRC32 checksum of the top 10 levels, so you detect a *wrong book* after applying each update and resubscribe. The Kraken checksum is pinned to Kraken's own published test vector. (Kraken is the default because Binance is unavailable from Canada and the US.) On the first live recording, about 40,000 consecutive updates all passed their checksums.
- **A forecast that's accurate and worthless.** On an hour of real BTC/USD order-book data (65,000 checksum-verified updates), the order-flow forecast predicts the direction of the next mid moves with AUC 0.92 and an information coefficient of 0.33 (t = 24), out of sample and with no leakage (giving it features from one event in the future adds nothing). But a no-model rule, "the last move continues", gets most of the way there (IC 0.27; in a 12-minute sample it tied). The mid steps in half-ticks as first one side of the book requotes and then the other, so most of the "prediction" is the second half of a step already under way. Either way it's worth about 0.03 bp per trade against fees of tens of basis points. The pipeline now prints that baseline beside every model, because beating a coin flip proves nothing here.
- **Replay with queue position**, so a backtest asks "would my order actually have filled, given the 400 shares queued ahead of it" instead of assuming a fill at mid.
- **Purged walk-forward research**, which drops training rows whose labels overlap the test window. Ordinary k-fold leaks here and inflates every metric.
- **FIX 4.4 session layer** over TCP: logon, heartbeats and test requests, sequence-gap detection with resend requests, gap fills for admin messages, possible-duplicate replays, and checksum-validated framing. Order messages go to the OMS and execution reports come back. It's hardened against hostile clients: no cross-client cancels, no session hijacking, bounded resends and queues.
- **Rust through PyO3**, with the same portable interface as the Python book, so the simulator and order-book backtester run on either engine unchanged.

```bash
cd engine
python test.py     # all self-checks
python bench.py    # throughput and tail latency
```

See `engine/README.md` for the benchmark methodology and the before/after numbers, including the profiling mistake that led to the 28x tail-latency fix.

## How it's tested

[GitHub Actions](.github/workflows/ci.yml) runs on every push:

- every engine self-check and 460+ endpoint checks, on deterministic synthetic data so a slow Yahoo can't fail the build;
- `cargo test` and the Python-vs-Rust differential test on 20,000 orders, on both Linux and Windows;
- the C++ port built with g++ under `-Werror` and with MSVC, its selftest, a 200,000-op replay under AddressSanitizer and UBSan, and the Python-vs-C++ differential test;
- a build of the PyO3 extension, a 20,000-op comparison through it (which fails, rather than skips, if the extension is missing), and the entire backend suite rerun with the Rust book underneath;
- on main, benchmarks for both engines and criterion, uploaded as artifacts rather than asserted, because shared runners make timing thresholds flaky.

Every module also carries its own self-check, including negative controls: the stylized-facts tests must *fail* on Gaussian noise, and the research critic must reject strategies found in pure noise.

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
| `backend/book_backtest.py` | Order-book backtests: next-open market orders against a modelled book, per-trade TCA, the timing/rounding/cost/capacity split |
| `backend/portfolio.py` | Multi-asset backtests with sizing schemes, VaR/CVaR, beta, factor exposure, risk contributions |
| `backend/market_sim.py` | The agent-based market, including the Avellaneda-Stoikov maker and its PnL decomposition |
| `backend/stylized.py` | Stylized-facts tests for real and simulated returns |
| `backend/forecast.py` | Order-flow forecast (walk-forward, purge gap, negative control) and the forecast-skewed market maker with markouts |
| `backend/exec_sim.py` | Works parent orders through the simulated market and scores them |
| `frontend/index.html` | The web UI: Arena, Lab, Analysis, Regime, Compare, Portfolio, and Engine tabs, built with Chart.js and vanilla JS |
| `backend/ai.py` | Turns a plain-English strategy description into validated rules, using your own API key or a local model |
| `backend/test_bullpen.py` | Plain-Python test suite, run with `python backend/test_bullpen.py` |
| `engine/` | The matching engine (Python, Rust, C++), FIX session layer, OMS, risk, execution algos, Kraken and Binance capture, queue-position replay, and research pipeline |
| `.github/workflows/ci.yml` | CI: Python checks, Rust tests and the differential test on Linux and Windows, the PyO3 build, benchmarks |

## Roadmap

1. ~~Backtesting engine and web UI~~ (done)
2. ~~Custom strategy rule language~~ (done)
3. ~~Live arena with leaderboard~~ (done)
4. ~~Free deploy on the Render free tier~~ (done)
5. ~~Parameter optimizer, walk-forward validation, and Monte Carlo simulation~~ (done)
6. ~~Market regime detector with explainability and regime-gated backtesting~~ (done)
7. ~~Candlestick charts, volume, and a multi-ticker correlation heatmap~~ (done)
8. ~~CI on every push, including the Rust engine and the differential test~~ (done)
9. ~~The Rust engine inside the app through PyO3~~ (done)
10. ~~Order-book backtests with transaction cost analysis~~ (done)
11. ~~Avellaneda-Stoikov market maker and stylized-facts validation of the simulator~~ (done)
12. ~~Portfolio layer: sizing, VaR/CVaR, factor exposure~~ (done)
13. ~~FIX session layer over TCP and repeatable benchmarks~~ (done)
14. ~~C++ port, differential-tested and sanitizer-clean~~ (done)
15. ~~Fix the simulator's autocorrelated returns at the root~~ (done)
16. ~~Order-flow forecast and a forecast-skewed market maker, scored with markouts~~ (done)
17. ~~Kraken capture with checksum verification~~ (done)
18. Record real Kraken books and run the forecast and stylized facts on them
19. Move the simulator's agents out of Python: the book is only ~48% of a run, so a faster book alone caps the end-to-end gain near 1.9x
