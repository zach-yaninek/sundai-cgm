"""personalize.py — learn one person from the meals they log.

The population model treats everyone with the same labs identically. But the same
meal produces iAUC from 6 to 253 across this cohort, and per-subject mean response
spans 13 to 112 — an 8.7x range. Some of that is captured by the lab panel; a lot
of it is not.

So once a user logs meals with real outcomes, we correct the population
prediction by their own observed bias:

    offset = mean(observed - predicted) * k / (k + 5)

The shrinkage matters more than the offset. One logged meal moves the correction
only 17% of the way, because one meal is mostly noise; by five it is at 50% and by
fifteen at 75%. Without it, a single unusual breakfast would swing every
subsequent prediction.

`learning_curve()` measures what this actually buys, leave-one-subject-out, and
writes it to `artifacts/learning_curve.json` — a real curve on held-out people
rather than a claim that the app "learns you".

    python personalize.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

import evaluate
import rung5_meal_risk as r5

ARTIFACTS = Path(__file__).parent / "artifacts"

# Shrinkage constant. 5 was chosen because the measured curve flattens around
# k=5-10: most of the available gain arrives by the fifth logged meal, so a
# constant that reaches half weight there tracks the evidence.
LAMBDA = 5.0

# Sweep points for the published curve.
CURVE_K = (0, 1, 2, 3, 5, 10, 15)

# A subject needs enough meals to both calibrate and be scored on the remainder.
MIN_MEALS = 25
HOLDOUT = 5


def shrinkage(k: int, lam: float = LAMBDA) -> float:
    """Weight given to a person's own history after ``k`` logged meals."""
    return 0.0 if k <= 0 else k / (k + lam)


def offset_from_residuals(residuals, lam: float = LAMBDA) -> float:
    """The correction to add to a population prediction, already shrunk.

    ``residuals`` are ``observed - predicted`` for meals the person has logged.
    Returns 0.0 for an empty history — an app that "personalises" before it has
    evidence is inventing the thing it claims to have learned.
    """
    residuals = np.asarray([r for r in residuals if r is not None and np.isfinite(r)],
                           dtype=float)
    if residuals.size == 0:
        return 0.0
    return float(residuals.mean() * shrinkage(residuals.size, lam))


def learning_curve(*, target: str = "iauc", with_glucose: bool = True,
                   seed: int = 0) -> pd.DataFrame:
    """Leave-one-subject-out: how does error fall as a person logs meals?

    For each held-out subject the population model is trained on everyone else,
    then ``k`` of that person's meals are revealed to compute the offset and the
    model is scored on the meals that remain. The calibration meals are excluded
    from scoring, or the curve would be measuring memorisation.
    """
    rng = np.random.default_rng(seed)
    df = r5.build_frame()
    X, _ = r5.build_features(df, with_glucose=with_glucose)
    y = df[target].to_numpy(dtype=float)
    groups = df["subject"].to_numpy()

    subjects = [s for s in np.unique(groups) if (groups == s).sum() >= MIN_MEALS]
    print(f"learning curve on {len(subjects)} subjects with >= {MIN_MEALS} meals "
          f"(target={target}, {'with' if with_glucose else 'no'} glucose)")

    rows = []
    for subject in subjects:
        train = groups != subject
        test = np.flatnonzero(groups == subject)

        model = xgb.XGBRegressor(**r5.REG_PARAMS)
        model.fit(X[train], y[train], verbose=False)
        predicted = model.predict(X.iloc[test])
        residuals = y[test] - predicted

        order = rng.permutation(len(test))
        for k in CURVE_K:
            if k > len(test) - HOLDOUT:
                continue
            calibration, evaluation = order[:k], order[k:]
            offset = offset_from_residuals(residuals[calibration]) if k else 0.0
            mae = float(np.abs(predicted[evaluation] + offset - y[test][evaluation]).mean())
            rows.append({"subject": int(subject), "k": k, "mae": mae})

    per_k = (pd.DataFrame(rows).groupby("k")
             .agg(mae=("mae", "mean"), n_subjects=("subject", "nunique"))
             .reset_index())
    baseline = float(per_k.loc[per_k.k == 0, "mae"].iloc[0])
    per_k["improvement"] = (baseline - per_k["mae"]).round(3)
    per_k["improvement_pct"] = (100 * per_k["improvement"] / baseline).round(1)
    per_k["shrinkage"] = [round(shrinkage(int(k)), 3) for k in per_k["k"]]
    return per_k


def main() -> None:
    curve = learning_curve()
    print()
    print(curve.round(3).to_string(index=False))

    best = curve.iloc[-1]
    print(f"\n{best.improvement_pct:.0f}% lower MAE after {int(best.k)} logged meals "
          f"({curve.mae.iloc[0]:.2f} -> {best.mae:.2f})")

    ARTIFACTS.mkdir(exist_ok=True)
    payload = {
        "target": "iauc",
        "units": "mg/dL*h",
        "lambda": LAMBDA,
        "method": "leave-one-subject-out; k meals revealed to compute a shrunk "
                  "offset, scored on that subject's remaining meals",
        "n_subjects": int(curve.n_subjects.max()),
        "min_meals_per_subject": MIN_MEALS,
        "points": [
            {"meals_logged": int(r.k), "mae": round(float(r.mae), 3),
             "improvement_pct": float(r.improvement_pct),
             "shrinkage": float(r.shrinkage)}
            for r in curve.itertuples()
        ],
    }
    (ARTIFACTS / "learning_curve.json").write_text(json.dumps(payload, indent=2))

    results_path = ARTIFACTS / "results.json"
    existing = json.loads(results_path.read_text()) if results_path.exists() else {}
    existing["personalization"] = {
        k: v for k, v in payload.items() if k != "points"
    } | {"points": payload["points"]}
    results_path.write_text(json.dumps(existing, indent=2))

    print(f"\nwrote {ARTIFACTS / 'learning_curve.json'}")


if __name__ == "__main__":
    main()
