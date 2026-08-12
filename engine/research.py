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
    from engine.replay import best_pair, features, ofi, replay
except ImportError:
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
def auc(y, score):
    """Rank-based AUC (Mann-Whitney U), tie-safe."""
    y = np.asarray(y)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    s_sorted = np.asarray(score)[order]
    ranks = np.empty(len(score), float)
    i = 0
    while i < len(s_sorted):          # average ranks within tie groups
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


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
        res = evaluate(X, y, mid, horizon=10, folds=3)
        assert res, "no folds evaluated"
        for r in res:
            assert 0.0 <= r["auc"] <= 1.0

    print(f"ok — purging, AUC, baseline, costs, pipeline all pass "
          f"({len(res)} folds on synthetic data)")


if __name__ == "__main__":
    demo()
