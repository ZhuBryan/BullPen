"""
book_backtest.py — a backtest whose orders fill against an order book.

run_backtest (backtest.py) fills every trade at the close, for the price of a
flat commission. That is an idealization: it pretends any amount of stock can
change hands at one printed price. Real orders pay for liquidity:

  * THE SPREAD. Buyers pay the ask, sellers get the bid. Even a tiny order
    loses half the bid-ask spread versus the mid price on every trade.
  * MARKET IMPACT. A large order eats the best price level, then the next,
    then the next. Each extra share is bought at a worse price. So cost per
    share GROWS with order size, which is why a strategy that works for $10k
    can fail for $10M.
  * CAPACITY. That growth is the capacity question every fund asks: how much
    money can this strategy run before costs eat the edge? slippage_curve()
    answers it directly.

What this module does, day by day:

  1. TIMING. A signal computed from day T's close is sent as an order at day
     T+1's OPEN. You cannot trade at the very close you used to compute the
     signal (by the time you know it, it has printed). This is a deliberate
     realism improvement over run_backtest's close-of-T fill. The result's
     `comparison` block separates that timing effect from the cost of the
     book itself, so the two can't be confused.
  2. LIQUIDITY MODEL. For each order we rebuild a synthetic limit order book
     around mid = Open[T+1] (integer cents). Half-spread comes from fills.py's
     calibrated range estimate (about one tick on a liquid large cap); depth
     is a fraction of trailing average volume, decaying away from the touch.
     Both use only data up to and including day T: no lookahead.
  3. EXECUTION. The strategy sends a MARKET order that walks that book
     (engine/orderbook.py, or the Rust engine when engine.fastbook exists).
     An order bigger than the book only partly fills; the remainder is worked
     again at the next day's open ("working a parent order").
  4. MARK TO MARKET at each day's Close, then compute run_backtest's metrics
     with identical definitions, and per-trade TCA via engine.execution.tca.

Run `python3 -m backend.book_backtest` for the self-check.
"""

import numpy as np
import pandas as pd

try:
    from engine.fastbook import Book
except ImportError:
    from engine.orderbook import Book
from engine.execution import Fill, tca
from backend.backtest import run_backtest, summary_metrics
from backend.fills import MIN_HALF_SPREAD, estimate_costs

# --- liquidity model constants ----------------------------------------------
TICK = 0.01            # US equities quote in whole cents; the book works in ticks
TOUCH_FRACTION = 0.0005  # shares at the best bid/ask = 0.05% of average daily
                         # volume. A rough stand-in for the liquidity a daily
                         # trader can take at the open without waiting.
LEVELS = 10            # price levels per side
LEVEL_SPACING_TICKS = 1  # gap between levels; 1 tick = a tightly quoted stock
DEPTH_DECAY = 0.8      # each level holds 80% of the one before it, so liquidity
                       # thins out away from the touch. Total depth per side is
                       # ~4.5x the touch, i.e. ~0.22% of ADV at the defaults.
                       # Sweeping all of it reaches 10 ticks past the touch, an
                       # average of ~3 ticks of impact per share: the same
                       # order of magnitude as fills.py's square-root law.
ADV_WINDOW = 20        # trailing days for average daily volume (~1 month)
MIN_TOUCH_SHARES = 100  # one round lot, so zero-volume bars still have a book




def _build_book(mid, half_frac, touch, levels):
    """A fresh book: `levels` bids below and asks above mid. Returns (book, half_ticks)."""
    book = Book()
    hs = max(1, int(round(half_frac * mid)))     # never tighter than 1 tick
    for k in range(levels):
        q = max(1, int(touch * DEPTH_DECAY ** k))
        off = hs + k * LEVEL_SPACING_TICKS
        book.limit("S", mid + off, q)
        if mid - off > 0:
            book.limit("B", mid - off, q)
    return book, hs


def _affordable(book, cash, commission_pct, levels):
    """Shares we can pay for (incl. commission) by walking the asks.

    Returns (qty, cash_limited). cash_limited=False means the book ran out
    first: the entry is not finished and continues tomorrow.
    """
    qty, left = 0, cash
    for px, q in book.depth("S", levels):
        unit = px * TICK * (1 + commission_pct)
        take = max(0, min(q, int(left // unit)))
        qty, left = qty + take, left - take * unit
        if take < q:
            return qty, True
    return qty, False


def _metrics(equity, starting_cash, trade_log, bh_return_pct):
    """backtest.summary_metrics on the ledger's equity, plus the trade stats."""
    ret = equity.pct_change()
    ret.iloc[0] = equity.iloc[0] / starting_cash - 1
    core, drawdown = summary_metrics(ret, starting_cash)
    wins = sum(1 for t in trade_log if t["return_pct"] > 0)
    return drawdown, {
        **core,
        "buy_hold_return_pct": bh_return_pct,
        "win_rate_pct": round(wins / len(trade_log) * 100 if trade_log else 0.0, 1),
        "num_trades": len(trade_log),
        # the ledger, not the compounded returns, is the source of truth
        "final_equity": round(float(equity.iloc[-1]), 2),
    }


def run(prices: pd.DataFrame, positions: pd.Series, starting_cash: float = 10_000.0,
        commission_pct: float = 0.001) -> dict:
    """Backtest `positions` (0/1 per day) with order-book fills at T+1's open.

    Returns run_backtest's keys plus fill_model, tca, tca_trades, comparison
    and ledger.
    """
    # The idealized close-fill run doubles as input validation (starting_cash)
    # and supplies dates / buy_hold / ohlc / volume, which do not depend on fills.
    ideal = run_backtest(prices, positions, starting_cash, commission_pct)

    idx, close = prices.index, prices["Close"]
    open_ = prices["Open"] if "Open" in prices.columns else close
    volume = (prices["Volume"] if "Volume" in prices.columns
              else pd.Series(0.0, index=idx)).fillna(0)
    # Liquidity for an order on day i is known by the close of day i-1:
    # shift(1) makes every number below use rows <= i-1 only.
    # Spread estimate known at the close of day i-1 (shift below); zero impact
    # here because impact is what walking the book itself produces.
    half = estimate_costs(prices, participation=0.0)
    half = half.shift(1).fillna(MIN_HALF_SPREAD)
    half = half.clip(lower=MIN_HALF_SPREAD)
    adv = volume.rolling(ADV_WINDOW, min_periods=1).mean().shift(1).fillna(0)
    touch = (adv * TOUCH_FRACTION).clip(lower=MIN_TOUCH_SHARES)
    # target[i] = what the signal at close i-1 wants us to hold during day i
    target = positions.reindex(idx).fillna(0).astype(int).shift(1).fillna(0).astype(int)

    cash, shares, entering = float(starting_cash), 0, False
    bought = sold = 0
    fees = 0.0
    equity, records, trade_log = [], [], []
    entry = None          # (date, buy notional, buy qty) of the open round trip
    exit_n = exit_q = 0.0
    for i, date in enumerate(idx):
        want = target.iloc[i]
        if want and (i == 0 or not target.iloc[i - 1]):
            entering = True                       # new entry signal: start buying
        side = "B" if want and entering else "S" if not want and shares > 0 else None
        if side:
            mid = int(round(open_.iloc[i] / TICK))
            book, hs = _build_book(mid, half.iloc[i], touch.iloc[i], LEVELS)
            if side == "B":
                qty, cash_limited = _affordable(book, cash, commission_pct, LEVELS)
                entering = not cash_limited and qty > 0   # book ran dry: keep working
            else:
                qty = shares
            if qty > 0:
                n0 = book.event_count()
                oid = book.market(side, qty)
                fills = [Fill(e[2], e[1] * TICK) for e in book.events_since(n0)
                         if e[0] == "trade" and e[3] == oid]
                filled = sum(f.qty for f in fills)
                notional = sum(f.qty * f.price for f in fills)
                fee = notional * commission_pct
                fees += fee
                if side == "B":
                    cash -= notional + fee
                    shares += filled
                    bought += filled
                    if entry is None:
                        entry = (date, 0.0, 0)
                    entry = (entry[0], entry[1] + notional, entry[2] + filled)
                else:
                    cash += notional - fee
                    shares -= filled
                    sold += filled
                    exit_n, exit_q = exit_n + notional, exit_q + filled
                # participation is measured against trailing ADV (known pre-trade),
                # so the record for day i depends on nothing printed on day i
                # except the open.
                r = tca(fills, side, qty, mid * TICK, market_volume=float(adv.iloc[i]) or None)
                records.append({
                    "date": str(date.date()), "side": side, "ordered": qty,
                    "filled": filled, "arrival_px": round(mid * TICK, 2),
                    "exec_vwap": round(r.exec_vwap, 4),
                    "notional_usd": round(notional, 4),
                    "slippage_bps": round(r.arrival_slippage_bps, 3),
                    "half_spread_bps": round(hs / mid * 1e4, 3),
                    "cost_usd": round(r.execution_cost, 2),
                    "commission_usd": round(fee, 2),
                    "fill_rate": round(r.fill_rate, 4),
                    "participation_of_adv": (round(r.participation_rate, 6)
                                      if r.participation_rate is not None else None),
                })
                if side == "S" and shares == 0 and entry is not None:
                    ep, xp = entry[1] / entry[2], exit_n / exit_q
                    trade_log.append({
                        "entry_date": str(entry[0].date()), "exit_date": str(date.date()),
                        "entry_price": round(ep, 2), "exit_price": round(xp, 2),
                        "return_pct": round((xp / ep - 1) * 100, 2)})
                    entry, exit_n, exit_q = None, 0.0, 0.0
        equity.append(cash + shares * float(close.iloc[i]))
    if entry is not None:                         # still holding at the end
        ep, xp = entry[1] / entry[2], float(close.iloc[-1])
        trade_log.append({"entry_date": str(entry[0].date()), "exit_date": "open",
                          "entry_price": round(ep, 2), "exit_price": round(xp, 2),
                          "return_pct": round((xp / ep - 1) * 100, 2)})

    equity = pd.Series(equity, index=idx)
    drawdown, metrics = _metrics(equity, starting_cash, trade_log,
                                 ideal["metrics"]["buy_hold_return_pct"])

    # --- aggregate TCA ---------------------------------------------------
    # cost_usd is the spread + impact paid versus the arrival mid (positive =
    # worse for us). Commission is reported separately, as desks do.
    traded = sum(r["notional_usd"] for r in records)
    cost = sum(r["cost_usd"] for r in records)
    worst = max(records, key=lambda r: r["slippage_bps"]) if records else None
    agg = {
        "num_orders": len(records),
        "traded_notional_usd": round(traded, 2),
        "total_cost_usd": round(cost, 2),
        "total_cost_bps": round(cost / traded * 1e4, 3) if traded else 0.0,
        "avg_slippage_bps": (round(float(np.mean([r["slippage_bps"] for r in records])), 3)
                             if records else 0.0),
        "commission_usd": round(fees, 2),
        "fill_rate": (round(sum(r["filled"] for r in records)
                            / sum(r["ordered"] for r in records), 4) if records else 1.0),
        "worst_trade": worst,
        "worst_trades": sorted(records, key=lambda r: -r["slippage_bps"])[:10],
    }

    book_ret, close_ret = metrics["total_return_pct"], ideal["metrics"]["total_return_pct"]
    gap = round(book_ret - close_ret, 2)
    # Split the gap into four causes that sum to it exactly. Each compares two
    # runs that differ in ONE thing:
    #   timing    next open instead of the signal day's close. Both runs are
    #             frictionless and hold fractional shares, like run_backtest.
    #   rounding  whole shares instead of fractional ones, still frictionless.
    #             Small for big accounts; large when a few shares of a pricey
    #             stock is the whole position (and it compounds over years).
    #   cost      spread and impact actually paid, in dollars (always <= 0).
    #   capacity  what is left: size the book could not absorb, so the account
    #             sat partly in cash or finished its entry days late. If every
    #             order filled completely it is labelled "other" instead: the
    #             interaction of rounding with costs. Paying the spread can
    #             leave you one share short; when the whole position is four
    #             shares of a stock that then rises 20x, that one share
    #             compounds into tens of points. Read rounding + other
    #             together for small accounts; for large ones both are tiny.
    # An earlier version had no rounding term and compared whole shares
    # against fractional ones inside "timing", which put about 200 points of
    # share-rounding drag on a 20-year, $1k run into the wrong bucket.
    open_frac = _frictionless_open_return(prices, positions, starting_cash, commission_pct,
                                          whole_shares=False)
    open_whole = _frictionless_open_return(prices, positions, starting_cash, commission_pct)
    timing = round(open_frac - close_ret, 2)
    rounding = round(open_whole - open_frac, 2)
    cost_pts = round(-cost / starting_cash * 100, 2)
    residual = round(gap - timing - rounding - cost_pts, 2)
    residual_label = "capacity" if agg["fill_rate"] < 1 else "other"
    share = round(-gap / close_ret * 100, 1) if close_ret > 0 and gap < 0 else None
    parts = (("timing (next open vs close)", timing), ("share rounding", rounding),
             ("execution cost", cost_pts),
             ("capacity (size the book could not absorb)" if residual_label == "capacity"
              else "other (rounding interacting with costs)", residual))
    comparison = {
        "close_fill_return_pct": close_ret,
        "book_fill_return_pct": book_ret,
        "return_gap_pct_points": gap,
        "frictionless_open_return_pct": round(open_frac, 2),
        "timing_effect_pct_points": timing,
        "rounding_effect_pct_points": rounding,
        "cost_effect_pct_points": cost_pts,
        "capacity_effect_pct_points": residual,
        "capacity_label": residual_label,
        "gap_share_of_close_return_pct": share,
        "close_fill_metrics": ideal["metrics"],
        "summary": (f"Order-book fills changed return by {gap:+} points vs close fills; "
                    "biggest driver: " + max(parts, key=lambda kv: abs(kv[1]))[0]),
        "note": f"Gap {gap:+} pts = timing {timing:+} (next open vs signal close) "
                f"+ rounding {rounding:+} (whole shares) "
                f"+ cost {cost_pts:+} (spread and impact paid) "
                f"+ {residual_label} {residual:+}"
                + (" (size the book could not absorb)." if residual_label == "capacity"
                   else " (share rounding interacting with costs: large only when a "
                        "few shares are the whole position)."),
    }

    out = dict(ideal)
    out.update({
        "equity": [round(float(v), 2) for v in equity],
        "drawdown_pct": [round(float(v) * 100, 2) for v in drawdown],
        "metrics": metrics,
        "trades": trade_log[-50:],
        "fill_model": "book",
        "tca": agg,
        # A 20-year, $100M run has thousands of child orders (a 22 MB response
        # in review). Send the most recent 500, plus the 10 worst by slippage,
        # which is what the UI shows.
        "tca_trades": records[-500:],
        "tca_trades_total": len(records),
        "comparison": comparison,
        "ledger": {"final_cash": round(cash, 6), "final_shares": shares,
                   "shares_bought": bought, "shares_sold": sold,
                   "commission_usd": round(fees, 6)},
        "liquidity_model": {"touch_fraction": TOUCH_FRACTION, "levels": LEVELS,
                            "level_spacing_ticks": LEVEL_SPACING_TICKS,
                            "depth_decay": DEPTH_DECAY, "adv_window": ADV_WINDOW,
                            "spread_model": "range (fills.py)"},
    })
    return out


def _frictionless_open_return(prices, positions, starting_cash, commission_pct,
                              whole_shares=True):
    """Total return % trading at T+1's open with no spread and unlimited depth.

    Same commission as run(). whole_shares=True sizes like run() (you can't
    buy 0.4 of a share); False sizes like run_backtest (fractional), which is
    what makes the timing comparison against close fills like-for-like.
    """
    open_ = prices["Open"] if "Open" in prices.columns else prices["Close"]
    target = positions.reindex(prices.index).fillna(0).astype(int).shift(1).fillna(0)
    cash, shares = float(starting_cash), 0.0
    for i in range(len(prices)):
        px = float(open_.iloc[i])
        if target.iloc[i] and shares == 0:
            shares = cash / (px * (1 + commission_pct))
            if whole_shares:
                shares = float(int(shares))
            cash -= shares * px * (1 + commission_pct)
        elif not target.iloc[i] and shares:
            cash += shares * px * (1 - commission_pct)
            shares = 0.0
    final = cash + shares * float(prices["Close"].iloc[-1])
    return (final / starting_cash - 1) * 100


def slippage_curve(prices, positions, cash_levels=(1e4, 1e5, 1e6, 1e7)) -> list:
    """Cost vs. account size: the capacity question.

    avg_slippage_bps is notional-weighted (total spread+impact cost / dollars
    traded), so a big parent order's small leftover child does not dilute it.
    """
    out = []
    for c in cash_levels:
        r = run(prices, positions, starting_cash=float(c))
        out.append({"starting_cash": float(c),
                    "avg_slippage_bps": r["tca"]["total_cost_bps"],
                    "mean_order_slippage_bps": r["tca"]["avg_slippage_bps"],
                    "fill_rate": r["tca"]["fill_rate"],
                    "num_orders": r["tca"]["num_orders"],
                    "total_return_pct": r["metrics"]["total_return_pct"],
                    "close_fill_return_pct": r["comparison"]["close_fill_return_pct"]})
    return out


def _synthetic(n=400, seed=7):
    """Deterministic offline OHLCV: a $50 random walk, ~1M shares/day."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    close = 50 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    open_ = np.r_[50.0, close[:-1]] * np.exp(rng.normal(0, 0.004, n))
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    vol = rng.integers(800_000, 1_200_000, n).astype(float)
    return pd.DataFrame({"Open": open_, "High": hi, "Low": lo, "Close": close,
                         "Volume": vol}, index=idx)


def demo():
    from backend.strategies import sma_crossover
    prices = _synthetic()
    pos = sma_crossover(prices, fast=10, slow=30)

    # 1. tiny account: every order fits at the touch, so slippage == half-spread
    tiny = run(prices, pos, starting_cash=1_000.0)
    assert tiny["tca_trades"], "demo strategy should trade"
    for r in tiny["tca_trades"]:
        assert abs(r["slippage_bps"] - r["half_spread_bps"]) < 1e-2, r

    # 2. capacity: cost per dollar never falls as the account grows
    curve = slippage_curve(prices, pos, cash_levels=(1e3, 1e4, 1e5, 1e6, 1e7))
    bps = [c["avg_slippage_bps"] for c in curve]
    # Tolerance 0.05bp: while every order still fits at the touch, accounts
    # differ only in how their dollars weight days with slightly different
    # spreads, which is noise, not a capacity effect.
    assert all(b >= a - 0.05 for a, b in zip(bps, bps[1:])), bps
    assert bps[-1] > bps[0], "large accounts must pay impact, not just spread"

    # 3. no lookahead. (a) every fill's arrival is that day's Open, one day
    # after the signal changed. (b) perturb everything from day k on (except
    # day k's Open, which is known when the order goes in): trades up to and
    # including day k are unchanged, so liquidity used only rows <= k-1.
    base = run(prices, pos, starting_cash=1e6)
    for r in base["tca_trades"]:
        d = pd.Timestamp(r["date"])
        assert r["arrival_px"] == round(round(prices.loc[d, "Open"] / TICK) * TICK, 2)
        i = prices.index.get_loc(d)
        # the order implements YESTERDAY's signal, never today's
        assert pos.iloc[i - 1] == (1 if r["side"] == "B" else 0), r
    flips = [prices.index[j + 1] for j in range(1, len(pos) - 1)
             if pos.iloc[j] != pos.iloc[j - 1]]
    dates = {pd.Timestamp(r["date"]) for r in base["tca_trades"]}
    assert flips and set(flips) <= dates, "each flip must trade at the next open"
    k = len(prices) // 2
    bent = prices.copy()
    rng = np.random.default_rng(1)
    for col in ("High", "Low", "Close", "Volume"):
        bent.iloc[k:, bent.columns.get_loc(col)] *= rng.uniform(0.5, 2.0, len(prices) - k)
    bent.iloc[k + 1:, bent.columns.get_loc("Open")] *= rng.uniform(0.5, 2.0, len(prices) - k - 1)
    bent["High"] = bent[["Open", "High", "Low", "Close"]].max(axis=1)
    bent["Low"] = bent[["Open", "High", "Low", "Close"]].min(axis=1)
    pert = run(bent, pos, starting_cash=1e6)
    cut = prices.index[k]
    early = lambda res: [r for r in res["tca_trades"] if pd.Timestamp(r["date"]) <= cut]
    assert early(base) and early(base) == early(pert), "liquidity peeked at the future"

    # 4. conservation: ledger == book fills, and cash reconciles to equity
    big = run(prices, pos, starting_cash=1e7)
    L, T = big["ledger"], big["tca_trades"]
    assert L["shares_bought"] == sum(r["filled"] for r in T if r["side"] == "B")
    assert L["shares_sold"] == sum(r["filled"] for r in T if r["side"] == "S")
    assert L["final_shares"] == L["shares_bought"] - L["shares_sold"]
    buys = sum(r["notional_usd"] for r in T if r["side"] == "B")
    sells = sum(r["notional_usd"] for r in T if r["side"] == "S")
    assert abs(1e7 - buys + sells - L["commission_usd"] - L["final_cash"]) < 0.01
    eq = L["final_cash"] + L["final_shares"] * prices["Close"].iloc[-1]
    assert abs(eq - big["metrics"]["final_equity"]) < 0.01
    assert big["tca"]["fill_rate"] <= 1 and min(r["fill_rate"] for r in T) > 0

    # 4b. the gap splits exactly into timing + rounding + cost + capacity;
    # paid cost is never a gain; timing doesn't depend on account size (it is
    # measured frictionless and fractional, so only the clock differs); a
    # fully filled account's remainder is labelled "other", not capacity,
    # while a size the book can't absorb shows up as capacity.
    for res in (tiny, big):
        c = res["comparison"]
        parts = ("timing", "rounding", "cost", "capacity")
        assert abs(sum(c[f"{k}_effect_pct_points"] for k in parts)
                   - c["return_gap_pct_points"]) <= 0.021, c
        assert c["cost_effect_pct_points"] <= 0, c
    assert tiny["comparison"]["timing_effect_pct_points"] == big["comparison"]["timing_effect_pct_points"]
    assert tiny["tca"]["fill_rate"] == 1 and tiny["comparison"]["capacity_label"] == "other"
    assert abs(tiny["comparison"]["capacity_effect_pct_points"]) < 0.5
    assert big["tca"]["fill_rate"] < 0.5 and big["comparison"]["capacity_label"] == "capacity"
    assert abs(big["comparison"]["capacity_effect_pct_points"]) > 1
    # the payload stays bounded; the UI's worst-trades table is server-side
    assert len(big["tca_trades"]) <= 500 and big["tca_trades_total"] >= len(big["tca_trades"])
    assert big["tca"]["worst_trades"][0]["slippage_bps"] == max(
        r["slippage_bps"] for r in big["tca_trades"]) or big["tca_trades_total"] > 500

    # 5. deterministic
    assert run(prices, pos, starting_cash=1e6) == base

    # 6. drop-in for the UI: a superset of run_backtest's keys and metrics
    ideal = run_backtest(prices, pos)
    assert set(ideal) <= set(tiny) and set(ideal["metrics"]) <= set(tiny["metrics"])
    assert len(tiny["equity"]) == len(ideal["equity"]) and tiny["fill_model"] == "book"

    print(f"ok — {len(tiny['tca_trades'])} orders; tiny-account slippage = half-spread "
          f"({tiny['tca']['avg_slippage_bps']}bps); capacity curve bps {bps}; "
          f"$10M: {big['comparison']['summary']}")


if __name__ == "__main__":
    demo()
