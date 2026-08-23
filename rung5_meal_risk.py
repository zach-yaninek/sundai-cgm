"""rung5_meal_risk.py — arbitrary meal + lab panel -> glucose response and risk.

Rung 4 one-hot encodes meal identity, so it can only score the 16 study dishes.
The app needs to score a meal somebody types in. This rung takes continuous
macros instead, which turns out to be the strongest model in the project:
cold-start MAE 28.0 / R² 0.307 against rung 1's 31.1.

Three heads share one feature builder:

    iauc        regression, mg/dL*h over 120 min   — the magnitude
    peak_abs    regression, mg/dL                  — what the UI shows
    exceeds     calibrated P(peak > 140 mg/dL)     — the flag

and each is trained in **two variants**, with and without a pre-meal glucose
reading, because the app has to work for someone with no CGM. Having a reading
lifts flag AUC from 0.841 to 0.888; `serve.py` picks the variant by what the
request actually carries rather than imputing a number and pretending.

    python rung5_meal_risk.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import cgm
import evaluate
import targets

ARTIFACTS = Path(__file__).parent / "artifacts"

# ADA's impaired-glucose-tolerance cut point. 45.8% of cohort meals cross it,
# across 40 of 45 subjects, so the classes are close to balanced.
THRESHOLD_MGDL = 140.0

# `amount_consumed` is deliberately absent. The published column mixes three
# incompatible scales -- 15 subjects record a percentage (50-100), 8 record small
# counts (0-9), and 16 have values up to 900 -- and the convention is per-subject,
# so the column partly encodes subject identity. Dropping it also *improves* both
# heads (MAE 28.30 -> 28.17, AUC 0.8871 -> 0.8884). See cgm.py NOTES item 10.
MACROS = ["carbs", "protein", "fat", "fiber", "calories"]
MEAL_TYPES = ["breakfast", "lunch", "dinner", "snack"]
LAB_FEATURES = [
    "age", "bmi", "body_weight", "height",
    "a1c_pdl_lab", "fasting_glu___pdl_lab", "insulin",
    "triglycerides", "cholesterol", "hdl", "non_hdl",
    "ldl_cal", "vldl_cal", "cho_hdl_ratio",
]
DERIVED = ["homa_ir"]

# See rung4_subject.py: self-reported race is a social category, not a
# physiological mechanism, and the lab panel already carries the physiology.
EXCLUDED_FEATURES = ["self_identify", "gender"]

REG_PARAMS = dict(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, min_child_weight=5,
    objective="reg:squarederror", random_state=0, n_jobs=4,
)
CLF_PARAMS = dict(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, min_child_weight=5,
    objective="binary:logistic", eval_metric="logloss", random_state=0, n_jobs=4,
)


# ---------------------------------------------------------------- features

def build_frame() -> pd.DataFrame:
    """Modelling rows joined to each subject's lab panel."""
    df = targets.modelling_set()
    bio = cgm.bio()[["subject"] + LAB_FEATURES].copy()
    bio["homa_ir"] = bio["fasting_glu___pdl_lab"] * bio["insulin"] / 405.0
    df = df.merge(bio, on="subject", how="left")
    df["peak_abs"] = df["baseline"] + df["peak_delta"]
    df["exceeds"] = (df["peak_abs"] > THRESHOLD_MGDL).astype(int)
    return df


def build_features(df: pd.DataFrame, *, with_glucose: bool) -> tuple[pd.DataFrame, dict]:
    """Feature matrix plus the spec needed to rebuild it identically at serving.

    ``baseline`` is the pre-meal glucose reading. It is a genuine input the user
    may not have, so it is a whole separate variant rather than an imputed
    column — imputing it would quietly hand every CGM-less user the median
    person's starting point and report the same confidence either way.
    """
    numeric = MACROS + LAB_FEATURES + DERIVED
    X = df[numeric].astype(float).copy()
    medians = X.median().to_dict()
    X = X.fillna(medians)

    for meal_type in MEAL_TYPES:
        X[f"mt_{meal_type}"] = (df["meal_type"] == meal_type).astype(float)

    if with_glucose:
        X["pre_meal_glucose"] = df["baseline"].astype(float)

    spec = {
        "columns": list(X.columns),
        "numeric": numeric,
        "meal_types": MEAL_TYPES,
        "impute_medians": {k: float(v) for k, v in medians.items()},
        "requires_pre_meal_glucose": with_glucose,
        "derived": {"homa_ir": "fasting_glu___pdl_lab * insulin / 405"},
        "excluded_features": EXCLUDED_FEATURES,
    }
    return X, spec


# ---------------------------------------------------------------- fitting

def _fit_regressor(X_tr, y_tr, X_te):
    model = xgb.XGBRegressor(**REG_PARAMS)
    model.fit(X_tr, y_tr, verbose=False)
    return model.predict(X_te)


def expected_calibration_error(prob, y, n_bins: int = 5) -> float:
    """Mean gap between what was predicted and what actually happened.

    The UI shows this probability as a number, so how well it is calibrated
    matters as much as how well the model ranks. AUC cannot see this at all.
    """
    prob, y = np.asarray(prob, dtype=float), np.asarray(y, dtype=float)
    idx = np.digitize(prob, np.linspace(0, 1, n_bins + 1)[1:-1])
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.sum():
            total += abs(prob[mask].mean() - y[mask].mean()) * mask.sum()
    return float(total / len(prob))


def _fit_calibrator(method: str, raw, y):
    """Fit one calibrator and return (predict_fn, serialisable knots)."""
    raw = np.asarray(raw, dtype=float)
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, y)
        knots = {"method": "isotonic",
                 "x": [float(v) for v in iso.X_thresholds_],
                 "y": [float(v) for v in iso.y_thresholds_]}
        return (lambda p: iso.predict(np.asarray(p, dtype=float))), knots
    lr = LogisticRegression(max_iter=1000).fit(raw.reshape(-1, 1), y)
    knots = {"method": "platt",
             "coef": float(lr.coef_[0][0]), "intercept": float(lr.intercept_[0])}
    return (lambda p: lr.predict_proba(np.asarray(p, dtype=float).reshape(-1, 1))[:, 1]), knots


def apply_calibrator(knots: dict, raw) -> np.ndarray:
    """Serving-side counterpart of :func:`_fit_calibrator`.

    Both forms serialise as plain numbers, so serving needs no pickle — a JSON
    array loads on any numpy without executing anything.
    """
    raw = np.asarray(raw, dtype=float)
    if knots["method"] == "isotonic":
        return np.clip(np.interp(raw, knots["x"], knots["y"]), 0.0, 1.0)
    z = knots["coef"] * raw + knots["intercept"]
    return 1.0 / (1.0 + np.exp(-z))


def _oof_probabilities(X, y, groups, method: str):
    """Out-of-fold probabilities, calibrated **within each fold**.

    Returns ``(calibrated, raw)``. The calibrated array is the honest estimate of
    serving performance; the raw one is what the serving calibrator is later
    fitted against, because it is the only sample of "scores this model gives to
    data it has not seen" that exists.

    Fitting a calibrator on all the data and then scoring the same rows would let
    it see the labels it is judged on, so each fold fits its own on an inner split.
    """
    calibrated = np.full(len(y), np.nan)
    raw_oof = np.full(len(y), np.nan)
    for train_idx, test_idx in evaluate.cold_folds(groups):
        inner = np.random.default_rng(0).permutation(len(train_idx))
        cut = int(len(inner) * 0.75)
        fit_idx, cal_idx = train_idx[inner[:cut]], train_idx[inner[cut:]]

        model = xgb.XGBClassifier(**CLF_PARAMS)
        model.fit(X.iloc[fit_idx], y[fit_idx], verbose=False)
        calibrate, _ = _fit_calibrator(method, model.predict_proba(X.iloc[cal_idx])[:, 1],
                                       y[cal_idx])
        raw_test = model.predict_proba(X.iloc[test_idx])[:, 1]
        raw_oof[test_idx] = raw_test
        calibrated[test_idx] = calibrate(raw_test)
    return calibrated, raw_oof


def select_calibrator(X, y, groups) -> tuple[str, dict, np.ndarray]:
    """Choose isotonic or Platt by held-out calibration error, not by default.

    They do not win in the same place. With a pre-meal glucose reading isotonic
    calibrates better (ECE 0.038 vs 0.050); without one it overfits the thinner
    signal and Platt is markedly better (0.040 vs 0.086) — and isotonic's failure
    there is the dangerous direction, understating risk in the middle band where
    it predicted 0.49 for meals that exceeded 67% of the time.
    """
    scored = {}
    for method in ("isotonic", "platt"):
        oof, raw_oof = _oof_probabilities(X, y, groups, method)
        scored[method] = {
            "oof": oof,
            "raw_oof": raw_oof,
            "ece": expected_calibration_error(oof, y),
            "brier": float(brier_score_loss(y, oof)),
            "auc": float(roc_auc_score(y, oof)),
        }
    best = min(scored, key=lambda m: scored[m]["ece"])
    for method, s in scored.items():
        mark = "  <- chosen" if method == best else ""
        print(f"    {method:<9} AUC {s['auc']:.3f}  Brier {s['brier']:.4f}  "
              f"ECE {s['ece']:.4f}{mark}")
    return best, scored, scored[best]["oof"], scored[best]["raw_oof"]


def _base_score(booster: xgb.Booster) -> str:
    """The intercept, as serialised. predict.py-style guard depends on this.

    xgboost >= 3.1 writes it as an array that older versions silently misread as
    0.5; recording it lets the serving side refuse rather than serve nonsense.
    """
    return json.loads(booster.save_config())["learner"]["learner_model_param"]["base_score"]


# ---------------------------------------------------------------------- main

def main() -> None:
    df = build_frame()
    groups = df["subject"].to_numpy()
    print(f"\nrung 5 — {len(df):,} meals · {df.subject.nunique()} subjects")
    print(f"exceeds {THRESHOLD_MGDL:.0f} mg/dL: {df.exceeds.mean():.1%} "
          f"({df.exceeds.sum()} meals, {df[df.exceeds == 1].subject.nunique()}/45 subjects)")

    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "models").mkdir(exist_ok=True)

    spec_out: dict = {
        "threshold_mgdl": THRESHOLD_MGDL,
        "target_units": {"iauc": "mg/dL*h over 120 min", "peak_abs": "mg/dL"},
        "xgboost_version": xgb.__version__,
        "variants": {},
    }
    reg_results = []

    for with_glucose in (True, False):
        variant = "with_glucose" if with_glucose else "no_glucose"
        X, spec = build_features(df, with_glucose=with_glucose)
        print(f"\n{'=' * 70}\nvariant: {variant}  ({X.shape[1]} features)\n{'=' * 70}")

        # --- regression heads -------------------------------------------------
        head_meta = {}
        for target in ("iauc", "peak_abs"):
            y = df[target].to_numpy(dtype=float)
            metrics, oof = evaluate.run(
                _fit_regressor, X, y, groups, regime="cold",
                name=f"{target} [{variant}]",
            )
            reg_results.append(metrics)

            final = xgb.XGBRegressor(**REG_PARAMS)
            final.fit(X, y, verbose=False)
            name = f"rung5_{target}_{variant}.json"
            final.get_booster().save_model(str(ARTIFACTS / "models" / name))
            head_meta[target] = {
                "model": name,
                "base_score": _base_score(final.get_booster()),
                "cv_mae": round(metrics["mae"], 3),
                "cv_r2": round(metrics["r2"], 4),
            }
            df[f"pred_{target}_{variant}"] = oof

        # --- classifier head --------------------------------------------------
        y = df["exceeds"].to_numpy()
        print(f"  exceeds_{THRESHOLD_MGDL:.0f} — calibrator selection:")
        method, scored, oof, raw_oof = select_calibrator(X, y, groups)
        df[f"pred_exceeds_{variant}"] = oof

        # Serving classifier: the booster is fit on everything, but the
        # calibrator is fitted on the RAW OUT-OF-FOLD scores against the true
        # labels. Those are the only sample we have of what this model does to
        # data it has not seen. Fitting the calibrator on the final model's own
        # in-sample scores would map a near-perfect training fit onto near-0/1
        # outputs and the UI would show 99% for almost every meal.
        final_clf = xgb.XGBClassifier(**CLF_PARAMS)
        final_clf.fit(X, y, verbose=False)
        _, knots = _fit_calibrator(method, raw_oof, y)

        clf_name = f"rung5_clf_{variant}.json"
        final_clf.get_booster().save_model(str(ARTIFACTS / "models" / clf_name))
        head_meta["exceeds"] = {
            "model": clf_name,
            "base_score": _base_score(final_clf.get_booster()),
            "calibrator": knots,
            "calibrator_method": method,
            "cv_auc": round(scored[method]["auc"], 4),
            "cv_brier": round(scored[method]["brier"], 4),
            "cv_ece": round(scored[method]["ece"], 4),
            "cv_average_precision": round(
                float(average_precision_score(y, oof)), 4),
            "calibrator_comparison": {
                m: {"ece": round(s["ece"], 4), "brier": round(s["brier"], 4),
                    "auc": round(s["auc"], 4)}
                for m, s in scored.items()
            },
        }

        # Cohort percentile lookup: a 101-point grid over out-of-fold
        # probabilities, so "higher than 80% of cohort meals" is a lookup rather
        # than something the server recomputes per request.
        spec["cohort_probability_quantiles"] = [
            float(q) for q in np.quantile(oof, np.linspace(0, 1, 101))
        ]
        spec["heads"] = head_meta
        spec_out["variants"][variant] = spec

    evaluate.report(reg_results, target="iauc / peak_abs")

    (ARTIFACTS / "rung5_spec.json").write_text(json.dumps(spec_out, indent=2))

    keep = ["subject", "timestamp", "meal_type", "carbs", "calories",
            "iauc", "peak_abs", "exceeds"]
    keep += [c for c in df.columns if c.startswith("pred_")]
    evaluate.upsert_predictions(ARTIFACTS / "predictions.parquet",
                                df[["subject", "timestamp"]].join(
                                    df[[c for c in df.columns if c.startswith("pred_")]]))
    df[keep].to_parquet(ARTIFACTS / "rung5_insample.parquet", index=False)

    results_path = ARTIFACTS / "results.json"
    existing = json.loads(results_path.read_text()) if results_path.exists() else {}
    existing["rung5"] = {
        "n": int(len(df)),
        "subjects": int(df.subject.nunique()),
        "threshold_mgdl": THRESHOLD_MGDL,
        "exceed_rate": round(float(df.exceeds.mean()), 4),
        "note": "Cold-start only. Two variants; serve.py picks by whether the "
                "request carries a pre-meal glucose reading.",
        "variants": {
            v: {"heads": {k: {kk: vv for kk, vv in h.items() if kk != "calibrator"}
                          for k, h in s["heads"].items()}}
            for v, s in spec_out["variants"].items()
        },
    }
    results_path.write_text(json.dumps(existing, indent=2))

    print(f"\nwrote {ARTIFACTS / 'rung5_spec.json'}")
    print(f"wrote 6 boosters to {ARTIFACTS / 'models'}")
    print(f"wrote {ARTIFACTS / 'rung5_insample.parquet'}")


if __name__ == "__main__":
    main()
