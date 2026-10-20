"""
Short-horizon prediction research: labels, purged walk-forward CV, baselines.

The models here are deliberately boring. Logistic regression first, trees only
if sklearn is installed. That is the point: on microstructure data the thing
that separates a real result from a fantasy is never the model, it is whether
the evaluation leaked. Three ways this file refuses to leak:

  1. NO LOOKAHEAD IN FEATURES. Features at index i are computed from book state
     at i. The label looks forward. They never touch.

  2. PURGED SPLITS. A label at index i depends on data through i+horizon. If i
     is in train and i+horizon reaches into test, that training row has already
     seen the test period. Standard k-fold happily does this and inflates every
     metric. We drop those rows (purge) plus a buffer after the test window
     (embargo), following the Lopez de Prado construction.

  3. WALK-FORWARD ONLY. Train is always strictly before test. No shuffling,
     because shuffling time series is training on the future to predict the past.

Reported metrics are AUC and cost-aware PnL, not accuracy. With 51/49 class
balance, accuracy of 0.51 looks like skill and is nothing.

Run `python research.py` for the self-check.
"""

import numpy as np

try:                                  # importable as a package or run directly
    from engine.capture import (KRAKEN_PRECISION, KrakenStream, Writer,
                                from_units, kraken_checksum)
    from engine.replay import best_pair, features, ofi, replay
except ImportError:
    from capture import KRAKEN_PRECISION, KrakenStream, Writer, from_units, kraken_checksum
    from replay import best_pair, features, ofi, replay


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------
FEATURE_NAMES = ["imbalance_1", "imbalance_n", "spread", "ofi", "micro_minus_mid"]


def build_dataset(path, horizon=20):
    """Walk a capture file into (X, y, mid) arrays.

    Label: did the microprice go UP over the next `horizon` book updates?
    Microprice rather than mid because mid only moves in whole ticks and is
    mostly flat, which makes the label almost pure noise at short horizons.

    Caveat: microprice depends on queue SIZES, and sizes mean-revert, so the
    current microprice-minus-mid partly predicts the future microprice even
    when the mid itself is a pure coin flip (the self-check's noise capture
    gets AUC around 0.65 here). That is not a tradeable edge. For a forecast
    of the price you can actually trade at, use orderflow_dataset(), whose
    label is the mid.
    """
    rows, micros = [], []
    prev_best = None

    for kind, _msg, book in replay(path):
        if kind != "book":
            continue
        f = features(book)
        bp = best_pair(book)
        if not f or bp is None:
            continue
        flow = ofi(prev_best, bp) if prev_best is not None else 0.0
        prev_best = bp
        rows.append([f["imbalance_1"], f["imbalance_n"], f["spread"],
                     flow, f["microprice"] - f["mid"]])
        micros.append(f["microprice"])

    X = np.asarray(rows, dtype=float)
    micro = np.asarray(micros, dtype=float)
    if len(micro) <= horizon:
        # All three must stay the same length; returning the full `micro` here
        # would break the len(X) == len(y) == len(mid) contract callers rely on.
        return X[:0], np.zeros(0), micro[:0]

    future = micro[horizon:]
    X, micro_now = X[:-horizon], micro[:-horizon]
    y = (future > micro_now).astype(int)
    return X, y, micro_now


# ---------------------------------------------------------------------------
# Purged walk-forward cross-validation
# ---------------------------------------------------------------------------
def purged_walk_forward(n, folds=4, horizon=20):
    """Yield (train_idx, test_idx): expanding window, train STRICTLY before test.

    Training data is only ever the past. That is stricter than Lopez de Prado's
    PurgedKFold, which also trains on data after the test block; PurgedKFold is
    a legitimate scheme, but it is not walk-forward and cannot be described as
    "the model only saw the past", which is the claim worth being able to make.

    Purging: a row at index i carries a label that looks `horizon` steps ahead,
    so it is only safe to train on when i + horizon < test_start. Rows failing
    that are dropped. No embargo is needed on the trailing side, because nothing
    after the test block is used for training in the first place.
    """
    fold = n // (folds + 1)
    if fold <= 0:
        return
    for k in range(1, folds + 1):
        test_start = k * fold
        test_end = min(test_start + fold, n)
        if test_end <= test_start:
            break
        test = np.arange(test_start, test_end)
        train = np.arange(0, test_start)
        yield train[train + horizon < test_start], test


# ---------------------------------------------------------------------------
# Baseline model — logistic regression, numpy only
# ---------------------------------------------------------------------------
class Logit:
    """ponytail: plain gradient descent. Swap for sklearn when you need
    calibration, class weights, or anything beyond a baseline."""

    def __init__(self, lr=0.5, epochs=400, l2=1e-3):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        Z = np.c_[np.ones(len(X)), (X - self.mu) / self.sd]
        self.w = np.zeros(Z.shape[1])
        for _ in range(self.epochs):
            p = 1 / (1 + np.exp(-Z @ self.w))
            grad = Z.T @ (p - y) / len(y) + self.l2 * np.r_[0, self.w[1:]]
            self.w -= self.lr * grad
        return self

    def predict_proba(self, X):
        Z = np.c_[np.ones(len(X)), (X - self.mu) / self.sd]
        return 1 / (1 + np.exp(-Z @ self.w))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _ranks(x):
    """1-based ranks, ties get their average rank (needed: mid moves are mostly 0)."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    # start index of each run of equal values, and its length
    starts = np.r_[0, np.flatnonzero(np.diff(xs)) + 1]
    lengths = np.diff(np.r_[starts, len(xs)])
    avg = starts + (lengths - 1) / 2 + 1
    ranks = np.empty(len(x), float)
    ranks[order] = np.repeat(avg, lengths)
    return ranks


def auc(y, score):
    """Rank-based AUC (Mann-Whitney U), tie-safe."""
    y = np.asarray(y)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = _ranks(score)
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def information_coefficient(score, outcome):
    """Spearman rank correlation of forecast vs realised move.

    The standard quant measure of forecast quality: 0 is no skill, and on
    short-horizon data an IC of a few hundredths is already meaningful. A
    constant forecast has no ranking, so its IC is defined as 0.
    """
    rs, ro = _ranks(score), _ranks(outcome)
    if rs.std() == 0 or ro.std() == 0:
        return 0.0
    return float(np.corrcoef(rs, ro)[0, 1])


def cost_aware_pnl(y, proba, move, cost_bps=1.0, threshold=0.55):
    """Trade only on confident signals, and charge for every trade.

    `move` is the actual price change per row in the same units as price. Cost
    is charged on entry, in basis points of price. Most microstructure edges are
    smaller than the spread, so this number going negative is the normal,
    honest outcome — not a bug to tune away.
    """
    proba, move = np.asarray(proba), np.asarray(move)
    side = np.where(proba > threshold, 1, np.where(proba < 1 - threshold, -1, 0))
    traded = side != 0
    gross = side * move
    cost = traded * (cost_bps / 1e4)
    net = gross - cost
    return {
        "trades": int(traded.sum()),
        "gross": float(gross.sum()),
        "net": float(net.sum()),
        "hit_rate": float((np.asarray(y)[traded] == (side[traded] > 0)).mean())
        if traded.any() else float("nan"),
    }


def evaluate(X, y, mid, horizon=20, folds=4, cost_bps=1.0):
    """Walk-forward evaluation against a majority-class baseline."""
    rel_move = np.zeros(len(mid))
    if len(mid) > horizon:
        rel_move[:-horizon] = (mid[horizon:] - mid[:-horizon]) / mid[:-horizon]

    out = []
    for train, test in purged_walk_forward(len(y), folds, horizon):
        if len(train) < 50 or len(np.unique(y[train])) < 2:
            continue
        model = Logit().fit(X[train], y[train])
        p = model.predict_proba(X[test])
        base = np.full(len(test), y[train].mean())   # predict the base rate
        out.append({
            "n_train": len(train), "n_test": len(test),
            "auc": auc(y[test], p),
            "auc_baseline": auc(y[test], base),
            "pnl": cost_aware_pnl(y[test], p, rel_move[test], cost_bps),
        })
    return out


# ---------------------------------------------------------------------------
# Short-horizon order-flow forecast: does the book predict the next mid move?
# ---------------------------------------------------------------------------
# The question every market maker and execution desk asks: given what the
# order book looks like RIGHT NOW, which way is the mid more likely to move
# over the next few events? Three features, all known at time t:
#
#   ofi_window   Order Flow Imbalance summed over the last `window` events.
#                Net buying pressure at the touch: bids growing or stepping up
#                and asks shrinking or stepping up are positive.
#   imbalance_1  Top-of-book size imbalance in [-1, 1]. A thin ask queue is
#                close to being eaten, which moves the mid up.
#   spread       Wide spreads mean less certainty and bigger, noisier moves.
#
# Target: sign of mid[t + h] - mid[t], over the next h BOOK EVENTS (not
# seconds; event time is the natural clock of an order book). Most of the
# time the mid does not move at all, so the model is trained and scored on the
# non-zero moves, while IC and PnL are measured on every row.
OF_FEATURES = ["ofi_window", "imbalance_1", "spread"]


def orderflow_dataset(path, horizon=10, window=10):
    """Capture file -> (X, move, mid). Row t uses only book states up to t.

    X[t]    = [ofi over events t-window+1..t, imbalance_1 at t, spread at t]
    move[t] = mid[t + horizon] - mid[t]   (the label looks forward; X never does)
    mid[t]  = mid at t, for converting moves to returns
    Works on Binance and Kraken captures alike (same record format).
    """
    flows, imb, spread, mids = [], [], [], []
    prev = None
    for kind, msg, book in replay(path):
        if kind != "book":
            continue
        bp = best_pair(book)
        if bp is None:
            continue
        if msg.get("type") == "snapshot":
            prev = None                   # a resync is a jump, not order flow
        (b, bq), (a, aq) = bp
        flows.append(ofi(prev, bp) if prev is not None else 0.0)
        prev = bp
        imb.append((bq - aq) / (bq + aq))
        spread.append(a - b)
        mids.append((a + b) / 2)

    mid = np.asarray(mids, float)
    if len(mid) <= horizon:
        return np.zeros((0, len(OF_FEATURES))), np.zeros(0), np.zeros(0)
    c = np.cumsum(flows)
    lagged = np.r_[np.zeros(window), c][:len(c)]          # c[t - window], or 0
    X = np.column_stack([c - lagged, imb, spread])
    move = mid[horizon:] - mid[:-horizon]
    return X[:-horizon], move, mid[:-horizon]


def _direction_metrics(score, move, rel_move, cost_bps, threshold, horizon):
    """score: P(up) per row (0.5 = no view). Metrics on one set of OOS rows."""
    nz = move != 0
    up = (move > 0).astype(int)
    called = nz & (score != 0.5)          # a non-zero move AND a directional view
    ic = information_coefficient(score - 0.5, move)
    return {
        "auc": auc(up[nz], score[nz]) if nz.any() else float("nan"),
        # hit rate: of the moves that happened, how often was the call right?
        "hit_rate": float(((score[called] > 0.5) == (move[called] > 0)).mean())
        if called.any() else float("nan"),
        "ic": ic,
        # Rough t-stat for the IC. Labels overlap (row t and t+1 share h-1
        # events of future), so there are only about n/h independent
        # observations, not n. Using n would make noise look significant.
        "ic_t": ic * np.sqrt(max(len(move) / max(horizon, 1), 1.0)),
        "pnl": cost_aware_pnl(up, score, rel_move, cost_bps, threshold),
    }


def evaluate_orderflow(X, move, mid, horizon=10, folds=4, cost_bps=1.0, threshold=0.55):
    """Purged walk-forward evaluation of the order-flow forecast vs baselines.

    For each fold: fit a logistic model of P(up | features) on the PAST rows
    whose mid actually moved, predict the test block. All test blocks are then
    pooled and scored together. Returned:

      model     the forecast
      majority  always predict the direction that was more common in training
                (the "I just follow the drift" baseline)
      zero      predict no move, never trade: AUC 0.5, IC 0, PnL exactly 0.
                The baseline any real strategy must beat after costs.
      continue  predict that the last `horizon` events' mid move continues.
                Needs no model and no fitting. On the first real Kraken
                capture (12 min of BTC/USD) it TIED the model out of sample
                (IC 0.327 vs 0.329); over a full hour the model pulled ahead
                (0.333 vs 0.270), so it learns a little beyond this rule: the mid moves in half-ticks, one side of
                the book requoting and then the other following, and 82% of
                moves continue the previous one, so most "predictability" is
                the second half of a step already under way. The model's
                higher AUC comes from rows where the mid hadn't just moved.
                A model that can't beat this has learned the book's
                mechanics, not anything about value.
    Each with AUC (on non-zero moves), hit_rate (non-zero moves), ic (all rows),
    and cost-aware pnl (cost_bps per trade, holding for `horizon` events).
    Note: PnL rows overlap (a position per row, each held h events), so treat
    the sum as a score for comparing models, not as achievable dollars.
    """
    X, move, mid = np.asarray(X, float), np.asarray(move, float), np.asarray(mid, float)
    rel = np.where(mid != 0, move / np.where(mid == 0, 1, mid), 0.0)
    idx, model_p, major_p = [], [], []
    for train, test in purged_walk_forward(len(move), folds, horizon):
        moved = train[move[train] != 0]
        ups = (move[moved] > 0).astype(int)
        if len(moved) < 50 or len(np.unique(ups)) < 2:
            continue
        model = Logit().fit(X[moved], ups)
        idx.append(test)
        model_p.append(model.predict_proba(X[test]))
        major_p.append(np.full(len(test), 1.0 if ups.mean() >= 0.5 else 0.0))
    if not idx:
        return {"folds": 0, "n_oos": 0}

    idx = np.concatenate(idx)
    m, r = move[idx], rel[idx]
    # Continuation baseline, from past mids only: mid[t] - mid[t - horizon].
    past = np.zeros(len(mid))
    past[horizon:] = mid[horizon:] - mid[:-horizon]
    cont_p = np.where(past[idx] > 0, 1.0, np.where(past[idx] < 0, 0.0, 0.5))
    args = (m, r, cost_bps, threshold, horizon)
    return {
        "folds": len(model_p), "n_oos": len(idx),
        "moved_frac": float((m != 0).mean()),
        "model": _direction_metrics(np.concatenate(model_p), *args),
        "majority": _direction_metrics(np.concatenate(major_p), *args),
        "zero": _direction_metrics(np.full(len(idx), 0.5), *args),
        "continue": _direction_metrics(cont_p, *args),
    }


def run(path, horizon=10, window=10, folds=4, cost_bps=1.0):
    """Everything on one capture file (Binance or Kraken). JSON-ready dict."""
    X, y, mid = build_dataset(path, horizon)
    Xo, move, mido = orderflow_dataset(path, horizon, window)
    return {"rows": int(len(Xo)), "horizon": horizon, "window": window,
            "microprice_logit": evaluate(X, y, mid, horizon, folds, cost_bps),
            "orderflow": evaluate_orderflow(Xo, move, mido, horizon, folds, cost_bps)}


# ---------------------------------------------------------------------------
# Synthetic Kraken capture (for the self-check)
# ---------------------------------------------------------------------------
def synthetic_kraken_capture(path, n=6000, signal=True, seed=0):
    """Write a Kraken-format capture by pushing fake v2 messages (with real
    checksums) through capture.KrakenStream, i.e. the exact conversion
    record_kraken() uses. Includes trades and one corrupted message with its
    resync, like a real recording.

    signal=True:  queue-depletion market. The mid moves when the best bid or
                  ask queue is exhausted, so a thin queue really does predict
                  the next move: imbalance and OFI carry genuine information.
    signal=False: pure noise. The mid moves by coin flip, independent of the
                  book, so no feature can predict it (the negative control).
    """
    rng = np.random.default_rng(seed)
    pp, qp = KRAKEN_PRECISION["BTC/USD"]
    depth, cent = 10, 10 ** (qp - 2)                  # qty unit: 0.01 BTC
    fresh = lambda: int(rng.integers(10, 150)) * cent  # 0.10 .. 1.50 BTC
    bids = {500000 - i: fresh() for i in range(depth)}         # 50000.0 down
    asks = {500001 + i: fresh() for i in range(depth)}         # 50000.1 up
    stream, w, ts = KrakenStream("BTC/USD", depth), Writer(path), [0.0]

    def emit(kind, cb=(), ca=(), corrupt=False):
        lv = lambda side, keys: [{"price": from_units(p, pp),
                                  "qty": from_units(side.get(p, 0), qp)} for p in keys]
        if kind == "snapshot":
            cb, ca = sorted(bids, reverse=True), sorted(asks)
        cs = kraken_checksum(sorted(bids.items(), reverse=True), sorted(asks.items()))
        msg = {"channel": "book", "type": kind,
               "data": [{"symbol": "BTC/USD", "bids": lv(bids, cb), "asks": lv(asks, ca),
                         "checksum": cs ^ 1 if corrupt else cs}]}
        ts[0] += 0.05
        rows, resub = stream.handle(msg, ts[0])
        for row in rows:
            w.write(row)
        if resub:                         # the generator plays the exchange too
            emit("snapshot")

    def shift(up):
        """Best ask (up) or best bid (down) is gone; the mid moves one tick."""
        b, a = max(bids), min(asks)
        if up:
            del asks[a]
            bids[a] = fresh()             # new best bid where the ask was
            asks[max(asks) + 1] = fresh()  # refill depth at the far end
            del bids[min(bids)]           # pushed out: NOT sent, client truncates
            emit("update", [a], [a, max(asks)])
        else:
            del bids[b]
            asks[b] = fresh()
            bids[min(bids) - 1] = fresh()
            del asks[max(asks)]
            emit("update", [b, min(bids)], [b])

    emit("snapshot")
    for i in range(n):
        if i == n // 2:                   # one corrupted message mid-recording
            bids[max(bids)] += cent
            emit("update", [max(bids)], [], corrupt=True)
            continue
        if signal:
            # one side's best queue gets a random order or cancel/trade
            side, best = (bids, max(bids)) if rng.random() < 0.5 else (asks, min(asks))
            delta = int(rng.normal(0, 40)) * cent
            if delta < 0 and rng.random() < 0.5:
                ts[0] += 0.001
                w.write({"type": "trade", "ts": ts[0], "exchange": "kraken",
                         "p": from_units(best, pp), "q": from_units(-delta, qp),
                         "m": side is bids, "T": None})
            if rng.random() < 0.01:       # news: a move the book did not foresee
                shift(up=rng.random() < 0.5)
            elif side[best] + delta <= 0:
                shift(up=side is asks)    # queue exhausted -> price moves
            else:
                side[best] += delta
                emit("update", [best] if side is bids else [], [best] if side is asks else [])
        else:
            if rng.random() < 0.08:
                shift(up=rng.random() < 0.5)
            else:                         # sizes re-drawn, unrelated to the future
                side, best = (bids, max(bids)) if rng.random() < 0.5 else (asks, min(asks))
                side[best] = fresh()
                emit("update", [best] if side is bids else [], [best] if side is asks else [])
    w.close()
    return stream


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
def demo():
    rng = np.random.default_rng(0)

    # ---- purging actually removes overlapping rows ----
    n, horizon = 500, 20
    seen_folds = 0
    for train, test in purged_walk_forward(n, folds=4, horizon=horizon):
        seen_folds += 1
        # the defining property: training data is strictly in the past
        assert train.max(initial=-1) < test[0], "trained on the future"
        # no training row's label window may reach into the test block
        overlap = train[train + horizon >= test[0]]
        assert len(overlap) == 0, f"purge failed: {overlap[:5]}"
        assert set(train) & set(test) == set(), "train/test overlap"
        # the newest usable training row sits exactly `horizon` before the test
        assert train.max(initial=-1) == test[0] - horizon - 1
    assert seen_folds == 4, f"expected 4 folds, got {seen_folds}"

    # ---- AUC sanity ----
    y = np.array([0, 0, 1, 1])
    assert auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5, "all ties -> 0.5"

    # ---- model learns a real signal ----
    X = rng.normal(size=(2000, 5))
    y_signal = (X[:, 0] + 0.3 * rng.normal(size=2000) > 0).astype(int)
    m = Logit().fit(X[:1500], y_signal[:1500])
    a = auc(y_signal[1500:], m.predict_proba(X[1500:]))
    assert a > 0.85, f"should learn an obvious signal, got AUC {a:.3f}"

    # ---- and finds nothing in noise (the important negative control) ----
    y_noise = rng.integers(0, 2, size=2000)
    m = Logit().fit(X[:1500], y_noise[:1500])
    a_noise = auc(y_noise[1500:], m.predict_proba(X[1500:]))
    assert 0.35 < a_noise < 0.65, f"pure noise should be ~0.5, got {a_noise:.3f}"

    # ---- costs must be able to erase a weak edge ----
    proba = np.full(200, 0.6)
    move = np.full(200, 0.00005)          # +0.5 bp per trade
    free = cost_aware_pnl(np.ones(200), proba, move, cost_bps=0.0)
    charged = cost_aware_pnl(np.ones(200), proba, move, cost_bps=1.0)
    assert free["net"] > 0 > charged["net"], "1bp cost should flip a 0.5bp edge"
    assert charged["trades"] == 200

    # no trades when nothing clears the confidence threshold
    flat = cost_aware_pnl(np.ones(10), np.full(10, 0.5), np.zeros(10))
    assert flat["trades"] == 0 and flat["net"] == 0.0

    # ---- end to end on a synthetic capture ----
    import gzip, json, tempfile
    with tempfile.TemporaryDirectory() as d:
        p = f"{d}/cap.jsonl.gz"
        rows = [{"type": "snapshot", "lastUpdateId": 0,
                 "bids": [["99", "10"]], "asks": [["101", "10"]]}]
        uid, bid_q = 1, 10.0
        for i in range(400):
            bid_q = max(1.0, bid_q + rng.normal())
            rows.append({"type": "diff", "U": uid, "u": uid + 1, "status": "applied",
                         "b": [["99", f"{bid_q:.2f}"]], "a": [["101", "10"]]})
            uid += 2
        with gzip.open(p, "wt") as fh:
            fh.write("\n".join(json.dumps(r) for r in rows))

        X, y, mid = build_dataset(p, horizon=10)
        assert len(X) == len(y) == len(mid) and len(X) > 100, f"thin dataset: {len(X)}"
        assert X.shape[1] == len(FEATURE_NAMES)
        assert not np.isnan(X).any(), "NaNs in features"
        res_ = evaluate(X, y, mid, horizon=10, folds=3)
        assert res_, "no folds evaluated"
        for r in res_:
            assert 0.0 <= r["auc"] <= 1.0

    # ---- tie-aware ranks and IC ----
    assert list(_ranks([3, 1, 1, 2])) == [4.0, 1.5, 1.5, 3.0]
    assert information_coefficient([1, 2, 3], [10, 20, 30]) == 1.0
    assert information_coefficient(np.zeros(5), np.arange(5)) == 0.0, "constant -> 0"

    # ---- Kraken-format captures: both pipelines run unchanged ----
    with tempfile.TemporaryDirectory() as d:
        sig, noise = f"{d}/kraken_signal.jsonl.gz", f"{d}/kraken_noise.jsonl.gz"
        st = synthetic_kraken_capture(sig, n=6000, signal=True, seed=1)
        assert st.mismatches == 1 and st.snapshots == 2, "corrupted message not resynced"
        synthetic_kraken_capture(noise, n=6000, signal=False, seed=1)
        with gzip.open(sig, "rt") as fh:
            kinds = {json.loads(l)["type"] for l in fh}
        assert {"snapshot", "diff", "trade", "checksum_mismatch"} <= kinds, kinds

        X, y, mid = build_dataset(sig, horizon=10)       # the original pipeline
        assert len(X) == len(y) == len(mid) > 5000
        kr = evaluate(X, y, mid, horizon=10, folds=4)
        assert len(kr) == 4 and all(0 <= r["auc"] <= 1 for r in kr)

        h = 10
        Xo, move, mido = orderflow_dataset(sig, horizon=h, window=10)
        assert Xo.shape == (len(move), len(OF_FEATURES)) and not np.isnan(Xo).any()
        res = evaluate_orderflow(Xo, move, mido, horizon=h, folds=4)
        m, maj, zero = res["model"], res["majority"], res["zero"]
        # a genuine order-flow edge is found, out of sample ...
        assert m["auc"] > 0.65 and m["ic_t"] > 3, f"missed a real signal: {m}"
        assert m["hit_rate"] > maj["hit_rate"] + 0.1, "should beat follow-the-drift"
        assert m["pnl"]["gross"] > 0
        # ... and costs still eat it: one BTC/USD tick is about 0.2 bp of price
        assert m["pnl"]["net"] < 0 < m["pnl"]["gross"], "1 bp/trade should exceed the edge"
        assert zero["pnl"]["trades"] == 0 and zero["pnl"]["net"] == 0.0
        assert zero["auc"] == 0.5 and zero["ic"] == 0.0

        # ---- negative controls: the same pipeline must find NOTHING ----
        # shuffled labels: features intact, link to the future destroyed
        shuf = evaluate_orderflow(Xo, rng.permutation(move), mido, horizon=h)["model"]
        # pure-noise market: mid is a coin flip the book knows nothing about
        Xn, mn, midn = orderflow_dataset(noise, horizon=h, window=10)
        nres = evaluate_orderflow(Xn, mn, midn, horizon=h)["model"]
        for name, r in (("shuffled labels", shuf), ("noise capture", nres)):
            assert abs(r["ic_t"]) < 2.5 and 0.4 < r["auc"] < 0.6, \
                f"{name} should show no skill, got {r}"

    print(f"ok — purging, AUC, IC, baselines, costs, pipeline all pass "
          f"({len(res_)} folds on synthetic data); kraken capture: order-flow "
          f"AUC {m['auc']:.2f} IC t {m['ic_t']:.1f}, controls IC t "
          f"{shuf['ic_t']:+.1f} / {nres['ic_t']:+.1f}")


def _print_report(out):
    of = out["orderflow"]
    print(f"{out['rows']} book events, horizon {out['horizon']} events, "
          f"OFI window {out['window']}")
    if not of.get("folds"):
        print("not enough data for walk-forward evaluation (record longer)")
        return
    print(f"order-flow forecast, {of['folds']} purged walk-forward folds, "
          f"{of['n_oos']} out-of-sample rows, mid moved in {of['moved_frac']:.1%}")
    print(f"{'':10}{'AUC':>7}{'hit':>7}{'IC':>8}{'IC t':>7}{'trades':>8}"
          f"{'gross bp/trade':>16}{'net bp/trade':>14}")
    for name in ("model", "continue", "majority", "zero"):
        r = of[name]
        n = max(r["pnl"]["trades"], 1)
        print(f"{name:10}{r['auc']:7.3f}{r['hit_rate']:7.3f}{r['ic']:8.3f}"
              f"{r['ic_t']:7.1f}{r['pnl']['trades']:8d}"
              f"{r['pnl']['gross'] * 1e4 / n:16.4f}{r['pnl']['net'] * 1e4 / n:14.4f}")
    print("|IC t| under ~2 is indistinguishable from noise. 'net' charges the "
          "cost per trade; real taker fees are far larger than 1 bp.")
    if of["continue"]["ic"] >= of["model"]["ic"]:
        print("NOTE: the no-model 'continue the last move' rule matches or beats the "
              "model, so its skill is book mechanics (half-tick mid steps), not signal.")
    aucs = [f["auc"] for f in out["microprice_logit"]]
    if aucs:
        print("microprice-label logit, per-fold AUC: "
              + " ".join(f"{a:.3f}" for a in aucs) + " (see build_dataset caveat)")


if __name__ == "__main__":
    import glob
    import os
    import sys

    # python research.py                                     self-check
    # python research.py data/kraken_BTCUSD_*.jsonl.gz [horizon] [cost_bps]
    #   A glob is expanded here (PowerShell does not) and the NEWEST match used.
    if len(sys.argv) > 1:
        matches = sorted(glob.glob(sys.argv[1]), key=os.path.getmtime)
        if not matches:
            sys.exit(f"no capture file matches {sys.argv[1]}")
        hz = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        cost = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
        print(f"capture: {matches[-1]}")
        _print_report(run(matches[-1], horizon=hz, cost_bps=cost))
    else:
        demo()
