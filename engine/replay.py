"""
Replay captured L2 data: rebuild the book, model realistic fills, extract features.

Why this file exists: the standard retail backtest assumes you get filled at the
mid price the instant you decide to trade. That is fiction. In reality you post a
limit order at the back of a FIFO queue and you are filled only when everyone
ahead of you is gone. Two strategies with identical signals can have opposite
PnL purely because of queue position, so modelling it is the whole game.

QUEUE MODEL (the one honest judgement call in here):
  * You join a price level behind everything already resting there:
        ahead = current visible quantity at that price
  * A TRADE against your level consumes the front of the queue. It eats `ahead`
    first, and only then fills you.
  * A CANCEL also shrinks the level, but you cannot tell from L2 data whether it
    was in front of you or behind you. We assume cancels are always BEHIND you,
    so they never improve your position.
  * Net effect: you only ever advance by real trades. This is deliberately
    pessimistic, which is the correct direction to be wrong in — it under-fills
    rather than inventing fills that never happened.

Run `python replay.py` for the self-check (no network, synthetic data).
"""

import gzip
import json

from capture import DepthStream

BUY, SELL = "B", "S"


# ---------------------------------------------------------------------------
# Reading a capture file back into book states
# ---------------------------------------------------------------------------
def replay(path):
    """Yield (kind, payload, book) for each recorded message, in order.

    kind is "book" after a depth update, or "trade" for a print. `book` is the
    live DepthStream, so callers see the state as of that moment.
    """
    stream = DepthStream()
    with gzip.open(path, "rt") as fh:
        for line in fh:
            msg = json.loads(line)
            kind = msg.get("type")

            if kind == "snapshot":
                stream.apply_snapshot(msg)
                yield "book", msg, stream
            elif kind == "diff":
                if stream.apply_event(msg) == "applied":
                    yield "book", msg, stream
            elif kind == "trade":
                yield "trade", msg, stream


# ---------------------------------------------------------------------------
# Fills
# ---------------------------------------------------------------------------
class RestingOrder:
    """A limit order you posted, tracked through the queue."""

    def __init__(self, side, price, qty, ahead):
        self.side, self.price, self.qty = side, float(price), int(qty)
        self.ahead = float(ahead)   # shares in front of you at your price
        self.filled = 0

    @property
    def remaining(self):
        return self.qty - self.filled

    def on_trade(self, trade_price, trade_qty, buyer_is_maker):
        """Apply one market print. Returns shares of YOURS that filled."""
        if self.remaining <= 0:
            return 0
        price, qty = float(trade_price), float(trade_qty)

        # buyer_is_maker=True  -> the aggressor was a SELLER, hitting bids
        # buyer_is_maker=False -> the aggressor was a BUYER, lifting asks
        if self.side == BUY:
            if not buyer_is_maker or price > self.price:
                return 0     # not hitting our side, or trading above our bid
        else:
            if buyer_is_maker or price < self.price:
                return 0

        # Queue eats those ahead of us first.
        eaten = min(self.ahead, qty)
        self.ahead -= eaten
        qty -= eaten
        if qty <= 0:
            return 0

        got = int(min(self.remaining, qty))
        self.filled += got
        return got


def queue_ahead(book, side, price):
    """Shares already resting at `price` — everything that beats you in line."""
    levels = book.bids if side == BUY else book.asks
    return float(levels.get(str(price), levels.get(f"{float(price):g}", 0)) or 0)


# ---------------------------------------------------------------------------
# Microstructure features
# ---------------------------------------------------------------------------
def top(book, n=5):
    bids = sorted(book.bids.items(), key=lambda kv: -float(kv[0]))[:n]
    asks = sorted(book.asks.items(), key=lambda kv: float(kv[0]))[:n]
    return ([(float(p), float(q)) for p, q in bids],
            [(float(p), float(q)) for p, q in asks])


def features(book, n=5):
    """Snapshot features. Returns {} when the book is one-sided."""
    bids, asks = top(book, n)
    if not bids or not asks:
        return {}
    (bp, bq), (ap, aq) = bids[0], asks[0]
    bid_depth = sum(q for _, q in bids)
    ask_depth = sum(q for _, q in asks)

    return {
        "mid": (bp + ap) / 2,
        "spread": ap - bp,
        # Microprice: mid weighted toward the thinner side. A bid with 100 lots
        # and an ask with 1 lot is much likelier to tick up than down, and the
        # plain mid is blind to that.
        "microprice": (bq * ap + aq * bp) / (bq + aq),
        # Top-of-book imbalance in [-1, 1]: +1 all bid, -1 all ask.
        "imbalance_1": (bq - aq) / (bq + aq),
        "imbalance_n": (bid_depth - ask_depth) / (bid_depth + ask_depth),
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
    }


def ofi(prev, curr):
    """Order Flow Imbalance between two consecutive book states.

    Cont/Kukanov/Stoikov: an order at the bid adds pressure if the bid price
    rose or held; it removes pressure if the bid price fell. Symmetric for asks.
    Signed size, not raw volume — this is the feature that consistently predicts
    short-horizon moves, so it is worth getting exactly right.
    """
    (pb, pbq), (pa, paq) = prev
    (cb, cbq), (ca, caq) = curr
    e = 0.0
    if cb >= pb:
        e += cbq
    if cb <= pb:
        e -= pbq
    if ca <= pa:
        e -= caq
    if ca >= pa:
        e += paq
    return e


def best_pair(book):
    """((bid_price, bid_qty), (ask_price, ask_qty)) or None if one-sided."""
    bids, asks = top(book, 1)
    if not bids or not asks:
        return None
    return bids[0], asks[0]


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
def demo():
    # ---- queue position: you do not fill until those ahead are gone ----
    o = RestingOrder(BUY, 100.0, 10, ahead=25)
    assert o.on_trade(100.0, 20, buyer_is_maker=True) == 0, "should still be queued"
    assert o.ahead == 5
    assert o.on_trade(100.0, 8, buyer_is_maker=True) == 3, "3 through after queue clears"
    assert o.filled == 3 and o.remaining == 7

    # a trade that does not touch your side changes nothing
    o = RestingOrder(BUY, 100.0, 10, ahead=0)
    assert o.on_trade(100.0, 5, buyer_is_maker=False) == 0, "buy aggressor lifts asks"
    assert o.on_trade(101.0, 5, buyer_is_maker=True) == 0, "trade above our bid"
    assert o.filled == 0
    assert o.on_trade(99.0, 4, buyer_is_maker=True) == 4, "sweeping through our price fills us"

    # sell side mirrors
    o = RestingOrder(SELL, 100.0, 10, ahead=2)
    assert o.on_trade(100.0, 1, buyer_is_maker=False) == 0
    assert o.on_trade(100.0, 6, buyer_is_maker=False) == 5
    assert o.on_trade(100.0, 99, buyer_is_maker=True) == 0, "sell aggressor hits bids"

    # never overfill
    o = RestingOrder(BUY, 100.0, 10, ahead=0)
    assert o.on_trade(100.0, 1000, buyer_is_maker=True) == 10
    assert o.on_trade(100.0, 1000, buyer_is_maker=True) == 0
    assert o.remaining == 0

    # ---- features ----
    b = DepthStream()
    b.apply_snapshot({"lastUpdateId": 1,
                      "bids": [["99", "10"], ["98", "5"]],
                      "asks": [["101", "2"], ["102", "1"]]})
    f = features(b)
    assert f["mid"] == 100.0 and f["spread"] == 2.0
    assert abs(f["imbalance_1"] - (10 - 2) / 12) < 1e-9
    # heavy bid, thin ask -> microprice sits above the mid
    assert f["microprice"] > f["mid"], "microprice should lean toward the thin side"
    assert features(DepthStream()) == {}, "one-sided book yields no features"

    # ---- OFI signs ----
    prev = ((99.0, 10.0), (101.0, 10.0))
    assert ofi(prev, ((99.0, 15.0), (101.0, 10.0))) > 0, "bid grew -> buy pressure"
    assert ofi(prev, ((99.0, 5.0), (101.0, 10.0))) < 0, "bid shrank -> sell pressure"
    assert ofi(prev, ((100.0, 10.0), (101.0, 10.0))) > 0, "bid stepped up -> buy pressure"
    assert ofi(prev, ((99.0, 10.0), (100.0, 10.0))) < 0, "ask stepped down -> sell pressure"
    assert ofi(prev, prev) == 0.0, "no change -> no flow"

    # ---- end-to-end replay of a synthetic capture ----
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = f"{d}/cap.jsonl.gz"
        rows = [
            {"type": "snapshot", "lastUpdateId": 100,
             "bids": [["99", "10"]], "asks": [["101", "10"]]},
            {"type": "diff", "U": 101, "u": 105,
             "b": [["99", "12"]], "a": [], "status": "applied"},
            {"type": "trade", "p": "99", "q": "4", "m": True, "T": 1},
            {"type": "diff", "U": 106, "u": 110,
             "b": [], "a": [["101", "0"], ["102", "7"]], "status": "applied"},
        ]
        with gzip.open(p, "wt") as fh:
            fh.write("\n".join(json.dumps(r) for r in rows))

        kinds, last = [], None
        trades = 0
        for kind, msg, book in replay(p):
            kinds.append(kind)
            if kind == "trade":
                trades += 1
            last = best_pair(book)
        assert kinds == ["book", "book", "trade", "book"], kinds
        assert trades == 1
        assert last == ((99.0, 12.0), (102.0, 7.0)), f"book wrong at end: {last}"

    print("ok — queue fills, features, OFI, and replay all pass")


if __name__ == "__main__":
    demo()
