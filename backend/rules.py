"""
rules.py — a tiny, SAFE language for writing trading rules as plain text.

Users type rules like:

    RSI(14) < 30 AND CLOSE > SMA(200)     (buy: oversold, but in an uptrend)
    SMA(20) < SMA(50)                     (sell: short-term trend turned down)

We NEVER eval()/exec() user input — that would let anyone run arbitrary code
on the server. Instead this module implements a real (but small) language
pipeline, the same way compilers do it:

  1. TOKENIZER  — chop the text into tokens: numbers, names, operators, parens
  2. PARSER     — recursive descent following the grammar in SPEC.md:

         expr       := orexpr
         orexpr     := andexpr ( OR andexpr )*      # OR binds loosest
         andexpr    := comparison ( AND comparison )*
         comparison := term ( < | > | <= | >= ) term
         term       := NUMBER | indicator | '(' expr ')'
         indicator  := CLOSE | SMA(n) | EMA(n) | RSI(n) | MOM(n)

  3. EVALUATOR  — walk the parsed tree once per rule, computing pandas Series
                  over the whole price history (vectorized = fast).

Finance notes on the indicators (all computed on daily closing prices):
  CLOSE   today's closing price
  SMA(n)  simple moving average — the plain mean of the last n closes
  EMA(n)  exponential moving average — like SMA but recent days count more
  RSI(n)  relative strength index, a 0-100 "how stretched is this move" dial
          (below ~30 = oversold, above ~70 = overbought)
  MOM(n)  momentum — the % return over the last n days (MOM(60) > 0 means
          "the stock is higher than it was 60 trading days ago")

While an indicator is still "warming up" (e.g. SMA(200) needs 200 days of
history, so its first 199 values are NaN), any comparison involving it is
simply FALSE — you can't act on a signal that doesn't exist yet.
"""

import operator
import re

import pandas as pd

MAX_RULE_LENGTH = 300      # plenty for any sane rule, blocks silly abuse
MIN_PERIOD, MAX_PERIOD = 1, 500


class RuleError(ValueError):
    """A rule failed to parse. The message is human-readable and points at
    the problem token, so the UI can show it directly to the user."""


# ---------------------------------------------------------------------------
# 1. Tokenizer
# ---------------------------------------------------------------------------
# Each pattern below is one token type. We try them in order at the current
# position; whatever matches becomes the next token.
_TOKEN_SPEC = [
    ("NUMBER", r"\d+(?:\.\d+)?"),   # 30, 0.5, 200
    ("IDENT",  r"[A-Za-z_]+"),      # CLOSE, SMA, AND, OR, ...
    ("OP",     r"<=|>=|<|>"),       # two-char ops first so '<=' isn't '<','='
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("SKIP",   r"\s+"),             # whitespace — matched, then thrown away
]
_MASTER_RE = re.compile("|".join(f"(?P<{name}>{pat})" for name, pat in _TOKEN_SPEC))


def _tokenize(text: str):
    """Turn the rule text into a list of (kind, text, position) tuples.
    `position` is the 0-based character index, used for error messages."""
    tokens = []
    i = 0
    while i < len(text):
        m = _MASTER_RE.match(text, i)
        if m is None:
            raise RuleError(
                f"Unexpected character '{text[i]}' at position {i + 1}. "
                "Allowed: numbers, CLOSE, SMA(n), EMA(n), RSI(n), MOM(n), "
                "AND, OR, < > <= >=, and parentheses."
            )
        if m.lastgroup != "SKIP":
            tokens.append((m.lastgroup, m.group(), i))
        i = m.end()
    return tokens


# ---------------------------------------------------------------------------
# 2. Parser (recursive descent — one small method per grammar rule)
# ---------------------------------------------------------------------------
# The parser produces a little tree of plain tuples ("AST nodes"):
#   ("num", 30.0)               a literal number
#   ("close",)                  today's close
#   ("ind", "SMA", 50)          an indicator with its period
#   ("cmp", "<", left, right)   a comparison  -> boolean
#   ("and", left, right)        both true     -> boolean
#   ("or", left, right)         either true   -> boolean
class _Parser:
    def __init__(self, text: str):
        self.text = text
        self.tokens = _tokenize(text)
        self.i = 0  # index of the next unread token

    # -- tiny helpers -----------------------------------------------------
    def _peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def _advance(self):
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def _keyword(self, word: str) -> bool:
        """Consume the keyword (AND/OR) if it's next; return True if we did."""
        tok = self._peek()
        if tok and tok[0] == "IDENT" and tok[1].upper() == word:
            self.i += 1
            return True
        return False

    @staticmethod
    def _describe(node) -> str:
        """Pretty-print a node for error messages, e.g. 'SMA(50)'."""
        kind = node[0]
        if kind == "num":
            return f"the number {node[1]:g}"
        if kind == "close":
            return "CLOSE"
        if kind == "ind":
            return f"{node[1]}({node[2]})"
        return "the expression"

    # -- grammar rules ------------------------------------------------------
    def parse(self):
        """expr := orexpr, then make sure nothing is left over."""
        if not self.tokens:
            raise RuleError("Rule is empty — try something like 'SMA(20) > SMA(50)'.")
        node = self._orexpr()
        leftover = self._peek()
        if leftover is not None:
            raise RuleError(
                f"Unexpected '{leftover[1]}' at position {leftover[2] + 1} — "
                "the rule already formed a complete condition before it."
            )
        return node

    def _orexpr(self):
        """orexpr := andexpr ( OR andexpr )*   — OR binds loosest."""
        node = self._andexpr()
        while self._keyword("OR"):
            node = ("or", node, self._andexpr())
        return node

    def _andexpr(self):
        """andexpr := comparison ( AND comparison )*  — AND binds tighter."""
        node = self._comparison()
        while self._keyword("AND"):
            node = ("and", node, self._comparison())
        return node

    def _comparison(self):
        """comparison := term ( < | > | <= | >= ) term
        A parenthesized sub-condition like (RSI(14) < 30) also counts as a
        complete comparison on its own."""
        left, left_is_bool = self._term()
        tok = self._peek()
        if tok and tok[0] == "OP":
            if left_is_bool:
                raise RuleError(
                    f"'{tok[1]}' at position {tok[2] + 1} can't compare a "
                    "condition in parentheses — comparisons work on numbers "
                    "and indicators, e.g. 'CLOSE > SMA(50)'."
                )
            op = self._advance()[1]
            right, right_is_bool = self._term()
            if right_is_bool:
                raise RuleError(
                    f"The right side of '{op}' is a condition in parentheses — "
                    "it must be a number or an indicator, e.g. 'RSI(14) {op} 30'."
                )
            return ("cmp", op, left, right)
        if left_is_bool:
            return left  # e.g. '(RSI(14) < 30)' — already a condition
        raise RuleError(
            f"Expected a comparison operator (<, >, <=, >=) after "
            f"{self._describe(left)}, e.g. 'CLOSE > SMA(50)'."
        )

    def _term(self):
        """term := NUMBER | indicator | '(' expr ')'
        Returns (node, is_boolean): a parenthesized expr is a full condition
        (boolean); numbers and indicators are numeric."""
        tok = self._peek()
        if tok is None:
            raise RuleError(
                "Rule ended unexpectedly — expected a number, CLOSE, "
                "SMA(n), EMA(n), RSI(n), MOM(n), or '('."
            )
        kind, text, pos = tok

        if kind == "NUMBER":
            self._advance()
            return ("num", float(text)), False

        if kind == "LPAREN":
            self._advance()
            node = self._orexpr()
            closing = self._peek()
            if closing is None or closing[0] != "RPAREN":
                raise RuleError(f"Missing closing ')' for the '(' at position {pos + 1}.")
            self._advance()
            return node, True

        if kind == "IDENT":
            name = text.upper()
            if name == "CLOSE":
                self._advance()
                return ("close",), False
            if name in ("SMA", "EMA", "RSI", "MOM"):
                self._advance()
                return ("ind", name, self._period(name)), False
            if name in ("AND", "OR"):
                raise RuleError(
                    f"Unexpected '{text}' at position {pos + 1} — expected a "
                    "number or indicator before it."
                )
            raise RuleError(
                f"Unknown name '{text}' at position {pos + 1}. "
                "Allowed: CLOSE, SMA(n), EMA(n), RSI(n), MOM(n)."
            )

        if kind == "RPAREN":
            raise RuleError(f"Unexpected ')' at position {pos + 1} — nothing to close.")

        raise RuleError(f"Unexpected '{text}' at position {pos + 1}.")

    def _period(self, name: str) -> int:
        """Parse the '(n)' after SMA/EMA/RSI/MOM. n must be an int 1-500."""
        tok = self._peek()
        if tok is None or tok[0] != "LPAREN":
            raise RuleError(f"{name} needs a period in parentheses, e.g. {name}(20).")
        self._advance()
        tok = self._peek()
        if tok is None or tok[0] != "NUMBER":
            got = f"'{tok[1]}'" if tok else "end of rule"
            raise RuleError(f"{name}(...) needs a whole number of days, got {got}.")
        if "." in tok[1]:
            raise RuleError(f"{name} period must be a whole number of days, got '{tok[1]}'.")
        n = int(tok[1])
        self._advance()
        if not (MIN_PERIOD <= n <= MAX_PERIOD):
            raise RuleError(
                f"{name} period must be between {MIN_PERIOD} and {MAX_PERIOD}, got {n}."
            )
        tok = self._peek()
        if tok is None or tok[0] != "RPAREN":
            raise RuleError(f"Missing ')' after {name}({n}.")
        self._advance()
        return n


def parse_rule(text: str):
    """Parse one rule string into an AST. Raises RuleError on bad input."""
    text = str(text or "")
    if len(text) > MAX_RULE_LENGTH:
        raise RuleError(f"Rule is too long (max {MAX_RULE_LENGTH} characters).")
    return _Parser(text).parse()


def validate_rules(buy_rule: str, sell_rule: str):
    """Check both rules. Returns (True, "") or (False, "human-readable msg")."""
    try:
        parse_rule(buy_rule)
    except RuleError as e:
        return False, f"Buy rule: {e}"
    try:
        parse_rule(sell_rule)
    except RuleError as e:
        return False, f"Sell rule: {e}"
    return True, ""


# ---------------------------------------------------------------------------
# 3. Evaluator
# ---------------------------------------------------------------------------
_OPS = {"<": operator.lt, ">": operator.gt, "<=": operator.le, ">=": operator.ge}


def _indicator_series(close: pd.Series, name: str, n: int) -> pd.Series:
    """Compute one indicator over the whole close-price history.
    All of them are NaN while they're still 'warming up' (not enough days)."""
    if name == "SMA":
        return close.rolling(n).mean()
    if name == "EMA":
        # min_periods=n keeps the first n-1 days NaN, like SMA, so the
        # 'warming up => comparison is false' rule applies consistently.
        return close.ewm(span=n, adjust=False, min_periods=n).mean()
    if name == "RSI":
        # Same construction as strategies.rsi_mean_reversion:
        # average gain vs average loss over the last n days, mapped to 0-100.
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(n).mean()
        loss = (-delta.clip(upper=0)).rolling(n).mean()
        rs = gain / loss.replace(0, 1e-9)
        return 100 - 100 / (1 + rs)
    if name == "MOM":
        # % return over the last n days: today's close vs n days ago.
        return close / close.shift(n) - 1
    raise RuleError(f"Unknown indicator '{name}'.")  # unreachable after parsing


def _eval(node, close: pd.Series):
    """Walk the AST. Numeric nodes return a float or a Series;
    boolean nodes (cmp/and/or) return a boolean Series."""
    kind = node[0]
    if kind == "num":
        return node[1]
    if kind == "close":
        return close
    if kind == "ind":
        return _indicator_series(close, node[1], node[2])
    if kind == "cmp":
        left = _eval(node[2], close)
        right = _eval(node[3], close)
        result = _OPS[node[1]](left, right)
        if isinstance(result, bool):
            # Both sides were plain numbers (e.g. '1 > 0') — broadcast the
            # constant answer across every day.
            result = pd.Series(result, index=close.index)
        # Pandas already treats 'NaN < x' as False, which is exactly the
        # "indicator still warming up => condition is false" behaviour we want.
        return result.astype(bool)
    if kind == "and":
        return _eval(node[1], close) & _eval(node[2], close)
    if kind == "or":
        return _eval(node[1], close) | _eval(node[2], close)
    raise RuleError(f"Bad rule node '{kind}'.")  # unreachable


def rule_signal(prices: pd.DataFrame, rule_text: str) -> pd.Series:
    """Evaluate one rule over price history -> boolean Series (True = fires)."""
    return _eval(parse_rule(rule_text), prices["Close"])


def rule_positions(prices: pd.DataFrame, buy_rule: str, sell_rule: str) -> pd.Series:
    """
    Turn a buy rule + sell rule into a 0/1 positions Series, same convention
    as the functions in strategies.py:
      - flat (0) and the buy rule fires  -> go long (1)
      - long (1) and the sell rule fires -> go to cash (0)
      - otherwise keep doing whatever we were doing (stateful!)
    The backtest engine handles the "trade tomorrow, not today" shift,
    so this is the position implied by TODAY's close.
    """
    buy = rule_signal(prices, buy_rule)
    sell = rule_signal(prices, sell_rule)

    positions, holding = [], 0
    for b, s in zip(buy.to_numpy(), sell.to_numpy()):
        if holding == 0 and b:
            holding = 1
        elif holding == 1 and s:
            holding = 0
        positions.append(holding)
    return pd.Series(positions, index=prices.index)
