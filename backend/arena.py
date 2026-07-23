"""
arena.py — the live paper-trading arena.

The idea: take ~2 years of REAL daily history and replay it like a horse
race, one trading day every couple of wall-clock seconds. A handful of agents
(each just a pair of DSL rules from rules.py) trade the same stock; the
frontend polls /api/arena/state and watches their equity curves diverge.

Design choices worth understanding:

  * NO background threads. The replay "clock" is computed lazily, only when
    somebody asks for state:
        day_index = start_offset + elapsed_seconds / seconds_per_day
    Wall-clock time does the animating for free.

  * start_offset = 60: the first 60 days of data are a warm-up period so
    indicators like SMA(50) already have history at day 0 of the race.
    Everyone sits in cash at $10,000 during warm-up — a fair starting line.

  * Equity series are PRECOMPUTED for the full data window the moment an
    agent is (re)built. That's cheap (vectorized pandas over ~500 rows), and
    it means every state request is just slicing lists — no math per poll.
    We only *reveal* the series up to the current day_index, so nobody can
    peek at the future.

  * Same no-lookahead convention as backtest.py: a signal computed from day
    T's close trades at day T+1's return. Commission 0.1% per position change.

  * When the replay reaches the last day, the season ends and a new one
    auto-starts (data re-fetched, same agents, equities reset) — again
    lazily, on the next state request.

  * Custom agents persist in data/agents.json so they survive restarts.
"""

import json
import os
import threading
import time
import uuid
from datetime import datetime, timedelta

from backend import rules
from backend.data import DATA_DIR, get_prices

STARTING_CASH = 10_000.0
COMMISSION_PCT = 0.001     # 0.1% of the portfolio per enter/exit, like backtest.py
SECONDS_PER_DAY = 2.0      # one trading day of history per 2 real seconds
START_OFFSET = 60          # warm-up days before the race starts
WINDOW_CALENDAR_DAYS = 730  # ~2 years of history per season

AGENTS_FILE = os.path.join(DATA_DIR, "agents.json")

# ---------------------------------------------------------------------------
# The 5 default personalities. Each is nothing more than two DSL rules —
# they run through the exact same parser/evaluator as user-created agents.
# ---------------------------------------------------------------------------
DEFAULT_AGENTS = [
    {
        "id": "warren",
        "name": "Warren",
        "emoji": "🧓",
        "tagline": "The trend is your friend",
        # Classic trend following: long while the 20-day average is above the 50-day.
        "buy_rule": "SMA(20) > SMA(50)",
        "sell_rule": "SMA(20) < SMA(50)",
    },
    {
        "id": "dipsy",
        "name": "Dipsy",
        "emoji": "🤿",
        "tagline": "Buys the dip, sells the rip",
        # Mean reversion: buy when RSI says oversold, sell when overbought.
        "buy_rule": "RSI(14) < 30",
        "sell_rule": "RSI(14) > 70",
    },
    {
        "id": "momo",
        "name": "Momo",
        "emoji": "🚀",
        "tagline": "Winners keep winning",
        # Momentum: hold only while the last ~3 months were positive.
        "buy_rule": "MOM(60) > 0",
        "sell_rule": "MOM(60) < 0",
    },
    {
        "id": "hodl",
        "name": "HODL",
        "emoji": "💎",
        "tagline": "Diamond hands, never sells",
        # Buy & hold: the buy rule is always true, the sell rule never is.
        "buy_rule": "CLOSE > 0",
        "sell_rule": "CLOSE < 0",
    },
    {
        "id": "flip",
        "name": "Flip",
        "emoji": "🪙",
        "tagline": "Flips on every short-term wiggle",
        # Hyperactive trend follower — trades a lot, pays a lot of commission.
        "buy_rule": "SMA(5) > SMA(10)",
        "sell_rule": "SMA(5) < SMA(10)",
    },
]
DEFAULT_IDS = {a["id"] for a in DEFAULT_AGENTS}


class Arena:
    """One arena = one ticker, one replayed season, one set of agents.
    The app uses a single global instance (see get_arena() below)."""

    def __init__(self, ticker: str = "SPY", seconds_per_day: float = SECONDS_PER_DAY,
                 agents_file: str = AGENTS_FILE, data_loader=None):
        # data_loader lets tests inject a synthetic DataFrame; the app
        # uses the real get_prices (yahoo -> cache -> sample fallback).
        self.lock = threading.RLock()
        self.seconds_per_day = float(seconds_per_day)
        self.agents_file = agents_file
        self.data_loader = data_loader or self._default_loader

        self.ticker = ticker.upper().strip()
        self.season = 0
        self.prices = None
        self.data_source = None
        self.dates = []        # list of "YYYY-MM-DD" strings, full window
        self.started_at = None  # wall-clock time the current season began
        self.agents = []        # runtime entries (spec + precomputed series)

        self.custom_specs = self._load_custom_specs()
        self._start_season(self.ticker)

    # ------------------------------------------------------------------
    # Data + persistence
    # ------------------------------------------------------------------
    def _default_loader(self, ticker: str):
        """Fetch ~2 years of daily prices ending today."""
        end = datetime.now() + timedelta(days=1)
        start = end - timedelta(days=WINDOW_CALENDAR_DAYS)
        return get_prices(ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    def _load_custom_specs(self):
        """Read saved custom agents from data/agents.json (if any)."""
        if not os.path.exists(self.agents_file):
            return []
        try:
            with open(self.agents_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
        except (json.JSONDecodeError, OSError):
            return []  # a corrupt file shouldn't take the arena down
        specs = []
        for item in saved if isinstance(saved, list) else []:
            # Skip anything that no longer parses (e.g. file edited by hand).
            ok, _ = rules.validate_rules(item.get("buy_rule", ""), item.get("sell_rule", ""))
            if ok and item.get("id") and item.get("name"):
                specs.append({
                    "id": str(item["id"]),
                    "name": str(item["name"])[:24],
                    "emoji": str(item.get("emoji") or "🤖"),
                    "buy_rule": str(item["buy_rule"]),
                    "sell_rule": str(item["sell_rule"]),
                    "created_at": str(item.get("created_at") or ""),
                })
        return specs

    def _save_custom_specs(self):
        os.makedirs(os.path.dirname(self.agents_file), exist_ok=True)
        with open(self.agents_file, "w", encoding="utf-8") as f:
            json.dump(self.custom_specs, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Season lifecycle
    # ------------------------------------------------------------------
    def _start_season(self, ticker: str):
        """(Re)load data and rebuild every agent's precomputed series."""
        ticker = ticker.upper().strip()
        prices, source = self.data_loader(ticker)
        if len(prices) <= START_OFFSET + 10:
            raise ValueError(
                f"Not enough data for '{ticker}' — need more than "
                f"{START_OFFSET + 10} trading days."
            )

        # Season numbering: a brand-new ticker starts at season 1;
        # replays of the same ticker keep counting up.
        if self.prices is None or ticker != self.ticker:
            self.season = 1
        else:
            self.season += 1

        self.ticker = ticker
        self.prices = prices
        self.data_source = source
        self.dates = [str(d.date()) for d in prices.index]
        self.started_at = time.time()

        # Rebuild all agents. Everyone (defaults + saved customs) joins at
        # the starting line: flat $10,000 through the warm-up period.
        self.agents = []
        for spec in DEFAULT_AGENTS:
            self.agents.append(self._build_agent(spec, is_default=True, join_day=START_OFFSET))
        for spec in self.custom_specs:
            self.agents.append(self._build_agent(spec, is_default=False, join_day=START_OFFSET))

    def _build_agent(self, spec: dict, is_default: bool, join_day: int) -> dict:
        """Precompute the FULL equity curve for one agent (then we just
        reveal slices of it as the replay clock advances)."""
        close = self.prices["Close"]

        # 1. What the rules say to hold each day (0/1, stateful enter/exit).
        raw_pos = rules.rule_positions(self.prices, spec["buy_rule"], spec["sell_rule"])

        # 2. Force cash before the agent's join day (warm-up for season
        #    starters, or the current day for agents created mid-season).
        raw_pos.iloc[:join_day] = 0

        # 3. No lookahead: a signal from day T's close trades on day T+1.
        held = raw_pos.shift(1).fillna(0)

        # 4. Daily P&L: stock return while long, minus 0.1% each enter/exit.
        stock_ret = close.pct_change().fillna(0)
        trades = held.diff().abs().fillna(held.iloc[0])
        strat_ret = held * stock_ret - trades * COMMISSION_PCT
        equity = STARTING_CASH * (1 + strat_ret).cumprod()

        return {
            "id": spec["id"],
            "name": spec["name"],
            "emoji": spec.get("emoji") or "🤖",
            "tagline": spec.get("tagline", "Custom fighter"),
            "is_default": is_default,
            "buy_rule": spec["buy_rule"],
            "sell_rule": spec["sell_rule"],
            "join_day": join_day,
            # Pre-rounded so building a state response is pure list slicing.
            "equity_series": [round(float(v), 2) for v in equity],
            "held": [int(v) for v in held],
        }

    def _current_day_index(self) -> int:
        """The lazy replay clock. Also handles season rollover: once the
        clock has moved PAST the final day (so the finale was visible for
        one tick), the season auto-restarts."""
        total = len(self.prices)
        raw = START_OFFSET + (time.time() - self.started_at) / self.seconds_per_day
        if raw >= total:  # finished + final day already shown -> new season
            self._start_season(self.ticker)
            raw = START_OFFSET
        return min(int(raw), total - 1)

    # ------------------------------------------------------------------
    # Public API used by main.py
    # ------------------------------------------------------------------
    def get_state(self) -> dict:
        """Everything the frontend needs, revealed only up to day_index."""
        with self.lock:
            day_index = self._current_day_index()

            entries = []
            for agent in self.agents:
                equity_now = agent["equity_series"][day_index]
                entries.append({
                    "id": agent["id"],
                    "name": agent["name"],
                    "emoji": agent["emoji"],
                    "tagline": agent["tagline"],
                    "is_default": agent["is_default"],
                    "buy_rule": agent["buy_rule"],
                    "sell_rule": agent["sell_rule"],
                    "equity_series": agent["equity_series"][:day_index + 1],
                    "equity": round(equity_now, 2),
                    "return_pct": round((equity_now / STARTING_CASH - 1) * 100, 2),
                    "position": "LONG" if agent["held"][day_index] else "CASH",
                })

            # Rank 1 = highest equity right now.
            entries.sort(key=lambda a: a["equity"], reverse=True)
            for rank, entry in enumerate(entries, start=1):
                entry["rank"] = rank

            # Candlestick data, revealed only up to the current replay day so
            # the chart never shows the future. Falls back to a flat candle if
            # a data source lacks Open/High/Low.
            df = self.prices.iloc[: day_index + 1]
            has_full = all(c in df.columns for c in ("Open", "High", "Low"))
            ohlc = []
            for i in range(len(df)):
                c = round(float(df["Close"].iloc[i]), 2)
                if has_full:
                    o = round(float(df["Open"].iloc[i]), 2)
                    h = round(float(df["High"].iloc[i]), 2)
                    lo = round(float(df["Low"].iloc[i]), 2)
                else:
                    o = h = lo = c
                ohlc.append({"o": o, "h": h, "l": lo, "c": c})
            volume = (
                [int(v) if v == v else 0 for v in df["Volume"]]
                if "Volume" in df.columns else [0] * len(df)
            )

            return {
                "ticker": self.ticker,
                "data_source": self.data_source,
                "day_index": day_index,
                "total_days": len(self.prices),
                "season": self.season,
                "date": self.dates[day_index],
                "dates": self.dates[:day_index + 1],
                "ohlc": ohlc,
                "volume": volume,
                "agents": entries,
            }

    def add_agent(self, name: str, emoji: str, buy_rule: str, sell_rule: str) -> dict:
        """Create a custom agent. It joins the race at the CURRENT day with a
        flat $10,000 history before that (no retroactive trading!).
        Raises ValueError with a user-facing message on bad input."""
        name = str(name or "").strip()
        if not name:
            raise ValueError("Name is required.")
        if len(name) > 24:
            raise ValueError("Name must be 24 characters or fewer.")
        emoji = str(emoji or "").strip() or "🤖"
        if len(emoji) > 8:
            raise ValueError("Emoji is too long.")

        ok, err = rules.validate_rules(buy_rule, sell_rule)
        if not ok:
            raise ValueError(err)

        with self.lock:
            day_index = self._current_day_index()
            spec = {
                "id": uuid.uuid4().hex[:8],
                "name": name,
                "emoji": emoji,
                "buy_rule": str(buy_rule),
                "sell_rule": str(sell_rule),
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            self.custom_specs.append(spec)
            self._save_custom_specs()
            self.agents.append(self._build_agent(spec, is_default=False, join_day=day_index))

            # Return the agent exactly as it appears in /api/arena/state
            # (including its current rank), so the frontend can use it as-is.
            state = self.get_state()
            return next(a for a in state["agents"] if a["id"] == spec["id"])

    def delete_agent(self, agent_id: str):
        """Remove a CUSTOM agent. Defaults are permanent residents."""
        if agent_id in DEFAULT_IDS:
            raise ValueError("Default agents can't be deleted.")
        with self.lock:
            if not any(s["id"] == agent_id for s in self.custom_specs):
                raise KeyError(f"No agent with id '{agent_id}'.")
            self.custom_specs = [s for s in self.custom_specs if s["id"] != agent_id]
            self.agents = [a for a in self.agents if a["id"] != agent_id]
            self._save_custom_specs()

    def reset(self, ticker: str = None) -> dict:
        """Start a fresh season immediately, optionally on a new ticker."""
        with self.lock:
            new_ticker = (ticker or self.ticker).upper().strip()
            if not new_ticker:
                raise ValueError("Ticker can't be empty.")
            self._start_season(new_ticker)
            return self.get_state()


# ---------------------------------------------------------------------------
# The single global arena, created lazily on the first arena request so the
# server starts instantly (the first data fetch can take a few seconds).
# ---------------------------------------------------------------------------
_ARENA = None
_ARENA_LOCK = threading.Lock()


def get_arena() -> Arena:
    global _ARENA
    with _ARENA_LOCK:
        if _ARENA is None:
            _ARENA = Arena()
        return _ARENA
