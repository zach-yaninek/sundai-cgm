"""evaluate.py — the honest bar every rung has to clear.

Two evaluation regimes, because they answer different product questions and a
number quoted from the wrong one is a false claim:

**cold** — `GroupKFold` on subject. The test subject never appears in training.
This is the webapp scenario: a stranger uploads a photo. The only available
floor is the global training mean.

**known** — stratified folds *within* each subject. Some of this person's other
meals are in training. This is the returning-user scenario, and it is the only
regime where "predict this person's own average response" is computable at all.

That distinction matters. Under `GroupKFold` there is no training data for the
held-out subject, so a per-subject-mean baseline cannot be formed — reporting
one alongside a cold-start score would be comparing against a number the cold
model was never allowed to know.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold

N_FOLDS = 5
N_BOOT = 2000
SEED = 0


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "n": int(len(y_true)),
    }


def _boot_mae_ci(y_true, y_pred, groups, *, n_boot=N_BOOT, seed=SEED):
    """Bootstrap the MAE by resampling *subjects*, not rows.

    Rows within a subject are correlated, so resampling rows would understate
    the interval — which is the same mistake as splitting rows at random.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    stats = []
    for _ in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in picked])
        stats.append(np.mean(np.abs(y_pred[idx] - y_true[idx])))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def cold_folds(groups, n_folds=N_FOLDS):
    """Subject-disjoint folds. A subject's meals are never split across sides."""
    n = min(n_folds, len(np.unique(groups)))
    return GroupKFold(n_splits=n).split(np.zeros(len(groups)), groups=groups)


def known_folds(groups, n_folds=N_FOLDS, seed=SEED):
    """Folds stratified *by* subject, so every subject appears on both sides."""
    return StratifiedKFold(n_splits=n_folds, shuffle=True,
                           random_state=seed).split(np.zeros(len(groups)), groups)


def run(fit_predict, X, y, groups, *, regime="cold", name="model", n_folds=N_FOLDS):
    """Cross-validate one model and return metrics plus out-of-fold predictions.

    ``fit_predict(X_tr, y_tr, X_te) -> y_hat`` keeps this agnostic to whether
    the estimator is XGBoost, a linear model, or a constant baseline.
    """
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    oof = np.full(len(y), np.nan)

    splitter = cold_folds(groups, n_folds) if regime == "cold" else known_folds(groups, n_folds)
    for train_idx, test_idx in splitter:
        X_tr = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        X_te = X.iloc[test_idx] if hasattr(X, "iloc") else X[test_idx]
        oof[test_idx] = fit_predict(X_tr, y[train_idx], X_te)

    done = ~np.isnan(oof)
    m = _metrics(y[done], oof[done])
    m["mae_ci"] = _boot_mae_ci(y[done], oof[done], groups[done])
    m["regime"] = regime
    m["name"] = name
    return m, oof


# ------------------------------------------------------------------ floors

def global_mean(X_tr, y_tr, X_te):
    """Predict the training mean for everyone. The trivial floor."""
    n = len(X_te) if not hasattr(X_te, "shape") else X_te.shape[0]
    return np.full(n, float(np.mean(y_tr)))


def subject_mean_factory(groups):
    """Predict each subject's own training mean — the *known-user* floor.

    Only meaningful under ``regime="known"``. Under cold folds the held-out
    subject has no training rows, so this degrades to the global mean and the
    comparison stops being informative; :func:`report` refuses that pairing.
    """
    groups = np.asarray(groups)

    def fit_predict(X_tr, y_tr, X_te):
        tr_idx = X_tr.index.to_numpy() if hasattr(X_tr, "index") else None
        te_idx = X_te.index.to_numpy() if hasattr(X_te, "index") else None
        if tr_idx is None or te_idx is None:
            return np.full(len(y_tr), float(np.mean(y_tr)))
        means = pd.Series(y_tr, index=groups[tr_idx]).groupby(level=0).mean()
        overall = float(np.mean(y_tr))
        return np.array([means.get(g, overall) for g in groups[te_idx]])

    return fit_predict


def report(results: list[dict], *, target="iauc") -> pd.DataFrame:
    """Format a list of metric dicts as the results table."""
    rows = []
    for r in results:
        lo, hi = r["mae_ci"]
        rows.append({
            "model": r["name"],
            "regime": r["regime"],
            "n": r["n"],
            "MAE": round(r["mae"], 2),
            "MAE 95% CI": f"[{lo:.1f}, {hi:.1f}]",
            "RMSE": round(r["rmse"], 2),
            "R2": round(r["r2"], 3),
        })
    df = pd.DataFrame(rows)
    print(f"\n{'=' * 78}\ntarget: {target}   (lower MAE is better; R2<=0 means "
          f"no better than predicting the mean)\n{'=' * 78}")
    print(df.to_string(index=False))
    return df
