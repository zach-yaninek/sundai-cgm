"""risk.py — score one meal for one person, from artifacts only.

The serving-side counterpart of `rung5_meal_risk.py`. No training code path, no
source data, no network. `recommend.py` and `serve.py` both go through here, so
there is exactly one implementation of how a feature vector gets built.

    import risk
    risk.score(labs, meal, pre_meal_glucose=104)

Variant selection is automatic: pass a pre-meal glucose reading and the stronger
model is used (AUC 0.881 vs 0.826); omit it and the weaker one is, with a wider
reported confidence band. The reading is never imputed — doing that would hand a
CGM-less user the median person's starting point and report the same confidence
as if they had measured it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ARTIFACTS = Path(__file__).parent / "artifacts"

_SPEC: dict | None = None
_BOOSTERS: dict = {}


class ArtifactError(RuntimeError):
    """Artifacts are missing, or inconsistent with the installed xgboost."""


def spec() -> dict:
    global _SPEC
    if _SPEC is None:
        path = ARTIFACTS / "rung5_spec.json"
        if not path.exists():
            raise ArtifactError(f"{path} missing — run `python rung5_meal_risk.py`")
        _SPEC = json.loads(path.read_text())
    return _SPEC


def _booster(variant: str, head: str):
    """Load one booster, verifying it deserialised the way it was written."""
    key = (variant, head)
    if key in _BOOSTERS:
        return _BOOSTERS[key]

    import xgboost as xgb

    meta = spec()["variants"][variant]["heads"][head]
    path = ARTIFACTS / "models" / meta["model"]
    if not path.exists():
        raise ArtifactError(f"{path} missing — run `python rung5_meal_risk.py`")

    booster = xgb.Booster()
    booster.load_model(str(path))

    # xgboost >= 3.1 serialises the learned intercept as an array; older versions
    # cannot parse it, silently fall back to 0.5, and every prediction comes back
    # wrong with no exception raised. Refuse rather than serve that.
    actual = json.loads(booster.save_config())["learner"]["learner_model_param"]["base_score"]
    if str(actual) != str(meta["base_score"]):
        raise ArtifactError(
            f"{meta['model']}: base_score is {actual!r} but the model was saved with "
            f"{meta['base_score']!r}. Your xgboost ({xgb.__version__}) cannot read this "
            f"model correctly — every prediction would be silently wrong. "
            f"Install xgboost>={spec()['xgboost_version'].split('.')[0]}.1 "
            f"(needs Python>=3.10)."
        )
    _BOOSTERS[key] = booster
    return booster


def variant_for(pre_meal_glucose: float | None) -> str:
    return "with_glucose" if pre_meal_glucose is not None else "no_glucose"


def lab_fields() -> list[str]:
    """Lab inputs the model uses. All optional; medians fill the gaps."""
    numeric = spec()["variants"]["no_glucose"]["numeric"]
    macros = {"carbs", "protein", "fat", "fiber", "calories"}
    return [c for c in numeric if c not in macros and c != "homa_ir"]


def build_row(labs: dict, meal: dict, pre_meal_glucose: float | None) -> tuple[np.ndarray, list[str]]:
    """One feature vector, in exactly the order the boosters expect.

    Returns ``(row, imputed)``. Order comes from the recorded spec rather than
    from whatever order a caller's dict happens to have — training/serving skew
    is invisible when it happens, because the model still returns a number.
    """
    variant = variant_for(pre_meal_glucose)
    sv = spec()["variants"][variant]

    known_labs = set(lab_fields())
    unexpected = set(labs) - known_labs - {"homa_ir"}
    if unexpected:
        raise ValueError(f"unknown lab field(s): {sorted(unexpected)}. "
                         f"Expected any of: {sorted(known_labs)}")

    values: dict = {k: v for k, v in labs.items() if v is not None}
    if "homa_ir" not in values:
        glu, ins = values.get("fasting_glu___pdl_lab"), values.get("insulin")
        if glu is not None and ins is not None:
            values["homa_ir"] = float(glu) * float(ins) / 405.0

    meal_type = meal.get("meal_type")
    if meal_type not in sv["meal_types"]:
        raise ValueError(f"meal_type must be one of {sv['meal_types']}, got {meal_type!r}")
    for key in ("carbs", "calories"):
        if meal.get(key) is None:
            raise ValueError(f"meal.{key} is required")

    for key in ("carbs", "protein", "fat", "fiber", "calories"):
        if meal.get(key) is not None:
            values[key] = float(meal[key])

    medians = sv["impute_medians"]
    row, imputed = [], []
    for col in sv["columns"]:
        if col.startswith("mt_"):
            row.append(1.0 if col == f"mt_{meal_type}" else 0.0)
        elif col == "pre_meal_glucose":
            row.append(float(pre_meal_glucose))
        elif col in values:
            row.append(float(values[col]))
        else:
            row.append(float(medians[col]))
            if col in known_labs or col == "homa_ir":
                imputed.append(col)
    return np.asarray(row, dtype=np.float32).reshape(1, -1), imputed


def _predict(variant: str, head: str, rows: np.ndarray) -> np.ndarray:
    import xgboost as xgb

    cols = spec()["variants"][variant]["columns"]
    return _booster(variant, head).predict(xgb.DMatrix(rows, feature_names=cols))


def apply_calibrator(knots: dict, raw) -> np.ndarray:
    """Mirror of the training-side calibrator. No pickle involved either way."""
    raw = np.asarray(raw, dtype=float)
    if knots["method"] == "isotonic":
        return np.clip(np.interp(raw, knots["x"], knots["y"]), 0.0, 1.0)
    return 1.0 / (1.0 + np.exp(-(knots["coef"] * raw + knots["intercept"])))


def cohort_percentile(probability: float, variant: str) -> int:
    """Where this probability sits among the cohort's own meals, 0-100."""
    grid = np.asarray(spec()["variants"][variant]["cohort_probability_quantiles"])
    return int(np.clip(np.searchsorted(grid, probability), 0, 100))


def confidence_band(imputed: list[str], pre_meal_glucose: float | None) -> str:
    """Widens with imputed labs and with no glucose reading.

    A wide band is a real statement about how little the model was told, and the
    UI is expected to make it look wide rather than only label it.
    """
    n_labs = len(lab_fields())
    missing = len([f for f in imputed if f != "homa_ir"])
    if missing >= n_labs * 0.6 or (missing >= n_labs * 0.35 and pre_meal_glucose is None):
        return "wide"
    if missing or pre_meal_glucose is None:
        return "moderate"
    return "narrow"


def score(labs: dict, meal: dict, pre_meal_glucose: float | None = None,
          *, offset: float = 0.0) -> dict:
    """Full assessment for one meal.

    ``offset`` is the personalisation correction from `personalize.py`, in
    mg/dL*h. It shifts the iAUC estimate and the peak proportionally, and is 0.0
    for a user with no logged history.
    """
    variant = variant_for(pre_meal_glucose)
    row, imputed = build_row(labs, meal, pre_meal_glucose)

    iauc = float(_predict(variant, "iauc", row)[0]) + offset
    iauc = max(0.0, iauc)
    peak = float(_predict(variant, "peak_abs", row)[0])

    # A personal offset on the excursion implies a matching shift in the peak.
    # 120 min of area maps to roughly its own height over the two-hour window.
    if offset:
        peak = max(peak + offset * 0.5, 0.0)

    knots = spec()["variants"][variant]["heads"]["exceeds"]["calibrator"]
    probability = float(apply_calibrator(knots, _predict(variant, "exceeds", row))[0])

    return {
        "probability": round(probability, 3),
        "threshold_mgdl": spec()["threshold_mgdl"],
        "cohort_percentile": cohort_percentile(probability, variant),
        "predicted_iauc": round(iauc, 1),
        "predicted_peak_mgdl": round(peak, 1),
        "variant": variant,
        "imputed_fields": imputed,
        "confidence_band": confidence_band(imputed, pre_meal_glucose),
    }


def performance() -> dict:
    """Cross-validated figures for the UI to display, per variant."""
    out = {}
    for variant, sv in spec()["variants"].items():
        heads = sv["heads"]
        out[variant] = {
            "auc": heads["exceeds"]["cv_auc"],
            "brier": heads["exceeds"]["cv_brier"],
            "ece": heads["exceeds"]["cv_ece"],
            "calibrator": heads["exceeds"]["calibrator_method"],
            "mae_iauc": heads["iauc"]["cv_mae"],
            "r2_iauc": heads["iauc"]["cv_r2"],
            "mae_peak": heads["peak_abs"]["cv_mae"],
            "r2_peak": heads["peak_abs"]["cv_r2"],
        }
    return out
