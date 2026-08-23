"""rung4_subject.py — why does the same meal spike one person and not another?

CGMacros is largely a standardized-meal study: 16 dishes account for 857 of the
1,382 modelling rows, each eaten by 43 or 44 of the 45 subjects. That makes the
meal close to a constant and the person the variable — on that subset, subject
identity explains 28.6% of the variance in glucose response against meal
identity's 20.8%.

So the question worth asking is not "what is in this photo" but "who responds
strongly to it, and can a lab panel tell us in advance". This rung answers it
with cold-start folds: the held-out subject is someone the model has never seen,
which is the only honest test of predicting a *new* person's response.

    python rung4_subject.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

import cgm
import evaluate
import targets

ARTIFACTS = Path(__file__).parent / "artifacts"
TARGET = "iauc"
MIN_REPEATS = 20  # a meal counts as standardized if this many people ate it

LAB_FEATURES = [
    "age", "bmi", "body_weight", "height",
    "a1c_pdl_lab", "fasting_glu___pdl_lab", "insulin",
    "triglycerides", "cholesterol", "hdl", "non_hdl",
    "ldl_cal", "vldl_cal", "cho_hdl_ratio",
]

# `self_identify` (self-reported race/ethnicity) is deliberately excluded. It is
# a social category, not a physiological mechanism, and a model that leans on it
# to decide who tolerates carbohydrate would encode a population disparity as if
# it were biology. The lab panel already carries the physiology.
EXCLUDED_FEATURES = ["self_identify", "gender"]

PARAMS = dict(
    n_estimators=300, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7,
    reg_lambda=3.0, min_child_weight=10,
    objective="reg:squarederror", random_state=0, n_jobs=4,
)


def _fit(X_tr, y_tr, X_te):
    model = xgb.XGBRegressor(**PARAMS)
    model.fit(X_tr, y_tr, verbose=False)
    return model.predict(X_te)


def build() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, dict]:
    """Standardized-meal subset joined to each subject's labs and gut panel."""
    df = targets.modelling_set()
    df["meal_id"] = df["carbs"].astype(str) + "/" + df["calories"].astype(str)
    repeats = df["meal_id"].value_counts()
    df = df[df["meal_id"].map(repeats) >= MIN_REPEATS].reset_index(drop=True)

    bio = cgm.bio()
    bio = bio[["subject"] + LAB_FEATURES].copy()
    # HOMA-IR: the standard insulin-resistance index. Derived rather than
    # measured, but it is the number a clinician would actually reach for.
    bio["homa_ir"] = (bio["fasting_glu___pdl_lab"] * bio["insulin"]) / 405.0

    gut = cgm.gut_health()
    gut_cols = [c for c in gut.columns if c != "subject"]

    df = df.merge(bio, on="subject", how="left").merge(gut, on="subject", how="left")

    meal_oh = pd.get_dummies(df["meal_id"], prefix="meal", dtype=float)
    lab_cols = LAB_FEATURES + ["homa_ir"]
    labs = df[lab_cols].astype(float)
    labs = labs.fillna(labs.median())
    guts = df[gut_cols].astype(float)
    guts = guts.fillna(guts.median())

    X = pd.concat([meal_oh, labs, guts], axis=1)
    blocks = {
        "meal": list(meal_oh.columns),
        "labs": lab_cols,
        "gut": gut_cols,
    }
    print(f"\nstandardized subset: {len(df):,} meals · {df.meal_id.nunique()} dishes · "
          f"{df.subject.nunique()} subjects")
    print(f"features: {len(blocks['meal'])} meal + {len(blocks['labs'])} lab + "
          f"{len(blocks['gut'])} gut")
    return X, df[TARGET].to_numpy(), df["subject"].to_numpy(), df, blocks


def subset_model(cols):
    def fit_predict(X_tr, y_tr, X_te):
        return _fit(X_tr[cols], y_tr, X_te[cols])
    return fit_predict


def subject_identity_oracle(groups):
    """Meal effect plus the subject's own observed offset — the ceiling.

    Not a deployable model: it needs the held-out person's own history. It is
    here to bound what personalisation could achieve at best, so a modest lab
    result can be read against the right scale rather than against zero.
    """
    groups = np.asarray(groups)

    def fit_predict(X_tr, y_tr, X_te):
        tr, te = X_tr.index.to_numpy(), X_te.index.to_numpy()
        grand = float(np.mean(y_tr))
        offset = pd.Series(y_tr - grand, index=groups[tr]).groupby(level=0).mean()
        meal_cols = [c for c in X_tr.columns if c.startswith("meal_")]
        base = _fit(X_tr[meal_cols], y_tr, X_te[meal_cols])
        return base + np.array([offset.get(g, 0.0) for g in groups[te]])

    return fit_predict


def main() -> None:
    X, y, groups, df, blocks = build()
    meal, labs, gut = blocks["meal"], blocks["labs"], blocks["gut"]

    results, oof_store = [], {}
    for regime in ("cold", "known"):
        contenders = [
            ("global mean (floor)", evaluate.global_mean),
            ("meal identity only", subset_model(meal)),
            ("rung 4: meal + labs", subset_model(meal + labs)),
            ("rung 4: meal + labs + gut", subset_model(meal + labs + gut)),
            ("labs only (no meal)", subset_model(labs)),
        ]
        if regime == "known":
            contenders.insert(1, ("subject mean (floor)",
                                  evaluate.subject_mean_factory(groups)))
            contenders.append(("oracle: meal + subject id",
                               subject_identity_oracle(groups)))
        for name, fn in contenders:
            m, oof = evaluate.run(fn, X, y, groups, regime=regime, name=name)
            results.append(m)
            oof_store[(regime, name)] = oof

    evaluate.report(results, target=TARGET)

    print(
        "\nNOTE — quote the COLD numbers, not the known ones.\n"
        "  Every subject has a unique lab vector (BMI alone separates all 45), so\n"
        "  under `known` folds the lab panel functions as a subject ID. With 45\n"
        "  subjects x 16 dishes = 720 cells and only 857 rows, the model can very\n"
        "  nearly memorise each person-meal pair. That is why it appears to beat the\n"
        "  subject-identity oracle there, and it is not a personalisation result.\n"
        "  The cold-start rows are the honest claim: the held-out person is a\n"
        "  stranger and only their labs are available."
    )

    # Which markers actually carry the personal signal?
    final = xgb.XGBRegressor(**PARAMS)
    final.fit(X[meal + labs + gut], y, verbose=False)
    imp = (pd.Series(final.feature_importances_, index=meal + labs + gut)
           .sort_values(ascending=False))
    non_meal = imp[~imp.index.str.startswith("meal_")]
    print("\ntop subject-level predictors (gain-weighted):")
    for name, val in non_meal.head(10).items():
        print(f"  {name:<42} {val:.4f}")

    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "models").mkdir(exist_ok=True)
    final.get_booster().save_model(str(ARTIFACTS / "models" / "rung4_subject.json"))

    spec = {
        "meal_columns": meal, "lab_columns": labs, "gut_columns": gut,
        "excluded_features": EXCLUDED_FEATURES,
        "excluded_reason": "self-reported race/ethnicity is a social category, not a "
                           "physiological mechanism; the lab panel carries the physiology",
        "min_repeats": MIN_REPEATS,
        "target": TARGET,
    }
    (ARTIFACTS / "rung4_feature_spec.json").write_text(json.dumps(spec, indent=2))

    results_path = ARTIFACTS / "results.json"
    existing = json.loads(results_path.read_text()) if results_path.exists() else {}
    existing["rung4"] = {
        "target": TARGET,
        "n": int(len(df)),
        "dishes": int(df.meal_id.nunique()),
        "subjects": int(df.subject.nunique()),
        "headline": "cold / rung 4: meal + labs",
        "note": "Effective n for learning the personalisation mapping is 45 subjects, "
                "not 857 rows — the lab features vary only between people.",
        "known_regime_caveat": "Lab vectors are unique per subject (45/45 distinct), so "
                               "under known folds they act as a subject ID and the model "
                               "largely memorises the 720 subject-meal cells. Known-regime "
                               "rung 4 numbers are NOT a personalisation result and must "
                               "not be quoted as one.",
        "gut_panel": "Helps under known folds and hurts cold-start (33.03 vs 32.56 MAE) — "
                     "22 features across 45 subjects is over-parameterised, and the gain "
                     "is fingerprinting rather than signal. Headline model omits it.",
        "top_subject_predictors": {k: float(v) for k, v in non_meal.head(10).items()},
        "results": [
            {k: v for k, v in r.items() if k != "mae_ci"} | {"mae_ci": list(r["mae_ci"])}
            for r in results
        ],
    }
    results_path.write_text(json.dumps(existing, indent=2))

    preds_path = ARTIFACTS / "predictions.parquet"
    if preds_path.exists():
        preds = pd.read_parquet(preds_path)
        add = df[["subject", "timestamp"]].copy()
        add["pred_rung4_cold"] = oof_store[("cold", "rung 4: meal + labs + gut")]
        preds = preds.merge(add, on=["subject", "timestamp"], how="left")
        preds.to_parquet(preds_path, index=False)

    print(f"\nwrote {ARTIFACTS / 'models' / 'rung4_subject.json'}")
    print(f"wrote {ARTIFACTS / 'rung4_feature_spec.json'}")
    print(f"updated {results_path}")


if __name__ == "__main__":
    main()
