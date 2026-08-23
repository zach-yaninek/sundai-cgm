"""personalize.py — learn one person from the meals they log.

The population model treats everyone with the same labs identically. But the same
meal produces iAUC from 6 to 253 across this cohort, and per-subject mean response
spans 13 to 112 — an 8.7x range. Some of that is captured by the lab panel; a lot
of it is not.

So once a user logs meals with real outcomes, we correct the population
prediction by what those outcomes say about them:

    corrected = (1 - w) * predicted + w * (intercept + slope * predicted)
    w         = k / (k + 5)

Below six logged meals the slope is pinned at 1 and this collapses to the flat
correction it began as, ``mean(observed - predicted) * w``. At six and above a
slope is fitted too, which lets the correction say "the model understates your
*large* responses" rather than only "the model runs low for you".

Six is measured, not chosen — see `personalize_compare.py`. Ungated, a
two-parameter fit is far worse at small k (28.00 against 25.77 at k=3), because
two parameters on three points fit the draw rather than the person.

The shrinkage matters more than either parameter. One logged meal moves the
correction only 17% of the way, because one meal is mostly noise; by five it is
at 50% and by fifteen at 75%. Without it, a single unusual breakfast would swing
every subsequent prediction.

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
# The maths lives in a dependency-free module so the API container does not have
# to import pandas, xgboost and the training code just to apply an offset.
from shrinkage import (LAMBDA, SLOPE_MIN_POINTS, fit_calibration,
                       offset_from_residuals, shrinkage)

ARTIFACTS = Path(__file__).parent / "artifacts"

__all__ = ["LAMBDA", "SLOPE_MIN_POINTS", "shrinkage", "offset_from_residuals",
           "fit_calibration", "learning_curve"]

# Sweep points for the published curve. 6 and 8 are here because that is where
# the slope engages: without them the chart jumps straight from 5 to 10 and hides
# the transition it exists to show.
CURVE_K = (0, 1, 2, 3, 5, 6, 8, 10, 15)

# A subject needs enough meals to both calibrate and be scored on the remainder.
MIN_MEALS = 25
HOLDOUT = 5

# Shuffles per subject. One is not enough: WHICH k meals happen to be revealed
# swings the result hard at small k, and a single draw produced a curve that rose
# from 24.38 at k=5 to 24.92 at k=6 before falling again - a bump at exactly the
# point the slope engages, which reads as a defect in the method rather than the
# noise it is. This is the app's headline chart; it should not wobble.
REPEATS = 7


def learning_curve(*, target: str = "iauc", with_glucose: bool = True,
                   seed: int = 0) -> pd.DataFrame:
    """Leave-one-subject-out: how does error fall as a person logs meals?

    For each held-out subject the population model is trained on everyone else,
    then ``k`` of that person's meals are revealed to fit their calibration and
    the model is scored on the meals that remain. The calibration meals are
    excluded from scoring, or the curve would be measuring memorisation.

    This measures what the app actually applies, which since v1.3.0 is a slope
    as well as an intercept once k reaches SLOPE_MIN_POINTS. Keeping this in step
    with the serving path is the point: a published curve for a correction the
    app no longer applies is a decorative line.
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

        for repeat in range(REPEATS):
            order = rng.permutation(len(test))
            for k in CURVE_K:
                if k > len(test) - HOLDOUT:
                    continue
                calibrate, evaluation = order[:k], order[k:]
                fitted = fit_calibration(predicted[calibrate], y[test][calibrate])
                adjusted = np.array([fitted.apply(float(v))
                                     for v in predicted[evaluation]])
                mae = float(np.abs(adjusted - y[test][evaluation]).mean())
                rows.append({"subject": int(subject), "k": k, "repeat": repeat,
                             "mae": mae,
                             "learned_slope": bool(fitted.learned_slope)})

    raw = pd.DataFrame(rows)
    by_subject = (raw.groupby(["k", "subject"])
                  .agg(mae=("mae", "mean"), learned_slope=("learned_slope", "any"))
                  .reset_index())
    per_k = (by_subject.groupby("k")
             .agg(mae=("mae", "mean"), n_subjects=("subject", "nunique"),
                  learned_slope=("learned_slope", "any"))
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
        "slope_min_points": SLOPE_MIN_POINTS,
        "method": "leave-one-subject-out; k meals revealed to fit a shrunk "
                  "calibration (intercept only below slope_min_points, slope and "
                  "intercept at or above it), scored on that subject's remaining "
                  "meals",
        "n_subjects": int(curve.n_subjects.max()),
        "min_meals_per_subject": MIN_MEALS,
        "repeats": REPEATS,
        "points": [
            {"meals_logged": int(r.k), "mae": round(float(r.mae), 3),
             "improvement_pct": float(r.improvement_pct),
             "shrinkage": float(r.shrinkage),
             "learned_slope": bool(r.learned_slope)}
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
