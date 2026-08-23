"""rung1_macros.py — logged macros -> glucose response.

What is knowable from a perfect food log? Every later rung has to beat this,
because a photo model that cannot match a hand-written food log has not earned
its place in the pipeline.

    python rung1_macros.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

import evaluate
import targets

ARTIFACTS = Path(__file__).parent / "artifacts"
TARGET = "iauc"

NUMERIC = ["carbs", "protein", "fat", "fiber", "calories", "amount_consumed", "baseline"]
CATEGORICAL = ["meal_type"]

PARAMS = dict(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, min_child_weight=5,
    objective="reg:squarederror", random_state=0, n_jobs=4,
)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Feature matrix plus the spec needed to rebuild it identically at serving.

    The spec is not a nicety. If serving builds columns in a different order,
    XGBoost still returns a number and it is quietly wrong.
    """
    X = df[NUMERIC].copy()
    for col in CATEGORICAL:
        dummies = pd.get_dummies(df[col].astype("string"), prefix=col, dtype=float)
        for cat in sorted(dummies.columns):
            X[cat] = dummies[cat]

    medians = X[NUMERIC].median().to_dict()
    X[NUMERIC] = X[NUMERIC].fillna(medians)

    spec = {
        "columns": list(X.columns),
        "numeric": NUMERIC,
        "categorical": CATEGORICAL,
        "meal_type_vocabulary": sorted(df["meal_type"].dropna().unique().tolist()),
        "impute_medians": {k: float(v) for k, v in medians.items()},
        "target": TARGET,
    }
    return X, spec


def fit_predict(X_tr, y_tr, X_te):
    model = xgb.XGBRegressor(**PARAMS)
    model.fit(X_tr, y_tr, verbose=False)
    return model.predict(X_te)


def carbs_only(X_tr, y_tr, X_te):
    """Carbohydrate alone — the classic predictor, and the reference to beat."""
    model = xgb.XGBRegressor(**{**PARAMS, "max_depth": 3})
    model.fit(X_tr[["carbs"]], y_tr, verbose=False)
    return model.predict(X_te[["carbs"]])


def main() -> None:
    df = targets.modelling_set()
    X, spec = build_features(df)
    y = df[TARGET].to_numpy()
    groups = df["subject"].to_numpy()

    results, oof_store = [], {}
    for regime in ("cold", "known"):
        contenders = [
            ("global mean (floor)", evaluate.global_mean),
            ("carbs only", carbs_only),
            ("rung 1: all macros", fit_predict),
        ]
        # A per-subject-mean floor only exists when the subject is in training.
        if regime == "known":
            contenders.insert(1, ("subject mean (floor)",
                                  evaluate.subject_mean_factory(groups)))
        for name, fn in contenders:
            m, oof = evaluate.run(fn, X, y, groups, regime=regime, name=name)
            results.append(m)
            oof_store[(regime, name)] = oof

    table = evaluate.report(results, target=TARGET)

    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "models").mkdir(exist_ok=True)

    final = xgb.XGBRegressor(**PARAMS)
    final.fit(X, y, verbose=False)
    final.get_booster().save_model(str(ARTIFACTS / "models" / "rung1_macros.json"))
    (ARTIFACTS / "feature_spec.json").write_text(json.dumps(spec, indent=2))

    preds = df[["subject", "timestamp", "meal_type", "carbs", "calories",
                "image_path", TARGET]].copy()
    for (regime, name), oof in oof_store.items():
        if name.startswith("rung 1"):
            preds[f"pred_rung1_{regime}"] = oof
    preds.to_parquet(ARTIFACTS / "predictions.parquet", index=False)

    payload = {
        "target": TARGET,
        "n": int(len(df)),
        "subjects": int(df.subject.nunique()),
        "sensor": cgm_sensor(),
        "results": [
            {k: v for k, v in r.items() if k != "mae_ci"} | {"mae_ci": list(r["mae_ci"])}
            for r in results
        ],
    }
    results_path = ARTIFACTS / "results.json"
    existing = json.loads(results_path.read_text()) if results_path.exists() else {}
    existing["rung1"] = payload
    results_path.write_text(json.dumps(existing, indent=2))

    print(f"\nwrote {ARTIFACTS / 'models' / 'rung1_macros.json'}")
    print(f"wrote {ARTIFACTS / 'feature_spec.json'} ({len(spec['columns'])} features)")
    print(f"wrote {ARTIFACTS / 'predictions.parquet'} ({len(preds):,} rows)")
    print(f"wrote {results_path}")


def cgm_sensor() -> str:
    import cgm
    return cgm.DEFAULT_SENSOR


if __name__ == "__main__":
    main()
