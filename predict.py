"""predict.py — the serving entry point.

    import predict
    predict.rank_meals({"fasting_glu___pdl_lab": 118, "insulin": 17.4, "a1c_pdl_lab": 6.5})
    predict.predict_response(labs, meal_id="66.0/712.0")

Given a person's fasting lab panel, which of the study's standardized dishes
will spike them, and by how much?

Everything is read from ``artifacts/`` — the trained booster, the feature spec,
the dish catalog and the observed response curves. There is **no training code
path here and no network call**, which is what keeps a deployment reproducible
and its container small.

Loading is deliberately strict: the feature vector is rebuilt from the spec's
recorded column order rather than from whatever order a caller's dict happens to
have. Training/serving skew is invisible when it happens — the model still
returns a confident number — so the order is pinned in data, not in convention.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACTS = Path(__file__).parent / "artifacts"

_SPEC = None
_MODEL = None
_CATALOG = None
_CURVES = None


class ArtifactError(RuntimeError):
    """Raised when the artifacts are missing or inconsistent with each other."""


# ---------------------------------------------------------------- loading

def spec() -> dict:
    global _SPEC
    if _SPEC is None:
        path = ARTIFACTS / "rung4_feature_spec.json"
        if not path.exists():
            raise ArtifactError(f"{path} missing — run `python rung4_subject.py` first")
        _SPEC = json.loads(path.read_text())
    return _SPEC


def model():
    global _MODEL
    if _MODEL is None:
        import xgboost as xgb

        path = ARTIFACTS / "models" / spec()["model"]
        if not path.exists():
            raise ArtifactError(f"{path} missing — run `python rung4_subject.py` first")
        booster = xgb.Booster()
        booster.load_model(str(path))
        n_model = booster.num_features()
        n_spec = len(spec()["columns"])
        if n_model != n_spec:
            raise ArtifactError(
                f"model expects {n_model} features but the spec lists {n_spec}. "
                "The artifacts are from different runs — re-run rung4_subject.py."
            )
        _MODEL = booster
    return _MODEL


def dish_catalog() -> pd.DataFrame:
    """The standardized dishes a caller can choose between, with observed spread."""
    global _CATALOG
    if _CATALOG is None:
        path = ARTIFACTS / "dish_catalog.parquet"
        if not path.exists():
            raise ArtifactError(f"{path} missing — run `python rung4_subject.py` first")
        _CATALOG = pd.read_parquet(path)
    return _CATALOG


def dish_curve(meal_id: str) -> pd.DataFrame:
    """The median *observed* response curve for a dish, on a 5-minute grid.

    This is an observed average across everyone who ate it, not a prediction for
    the caller. :func:`predict_response` scales it to the predicted iAUC so a
    plot has a plausible shape, and labels it as such.
    """
    global _CURVES
    if _CURVES is None:
        path = ARTIFACTS / "dish_curves.parquet"
        if not path.exists():
            raise ArtifactError(f"{path} missing — run `python rung4_subject.py` first")
        _CURVES = pd.read_parquet(path)
    return _CURVES[_CURVES["meal_id"] == meal_id].sort_values("minute")


# ---------------------------------------------------------------- features

def build_row(labs: dict, meal_id: str) -> np.ndarray:
    """Assemble one feature vector in exactly the order the booster expects.

    Missing labs fall back to the training medians recorded in the spec, so a
    partial panel degrades gracefully instead of erroring — but
    :func:`predict_response` reports which fields were filled in, because a
    prediction resting on six imputed values deserves to be visibly weaker.
    """
    s = spec()
    known = set(s["lab_columns"])
    unexpected = set(labs) - known - {"homa_ir"}
    if unexpected:
        raise ValueError(
            f"unknown lab field(s): {sorted(unexpected)}. "
            f"Expected any of: {sorted(known)}"
        )

    values = dict(labs)
    # HOMA-IR is derived, not asked for — compute it when its inputs are present
    # so a caller supplying a normal panel does not have to know the formula.
    if "homa_ir" not in values:
        glu, ins = values.get("fasting_glu___pdl_lab"), values.get("insulin")
        if glu is not None and ins is not None:
            values["homa_ir"] = float(glu) * float(ins) / 405.0

    if meal_id not in set(dish_catalog()["meal_id"]):
        raise ValueError(
            f"unknown meal_id {meal_id!r}. Choose one of "
            f"{sorted(dish_catalog()['meal_id'])}"
        )

    medians = s["impute_medians"]
    row, imputed = [], []
    for col in s["columns"]:
        if col.startswith("meal_"):
            row.append(1.0 if col == f"meal_{meal_id}" else 0.0)
        elif col in values and values[col] is not None:
            row.append(float(values[col]))
        else:
            row.append(float(medians[col]))
            imputed.append(col)
    return np.asarray(row, dtype=np.float32).reshape(1, -1), imputed


def _predict_raw(rows: np.ndarray) -> np.ndarray:
    """Predict, clipped at zero.

    iAUC is the area of the curve *above* its own baseline, so it cannot be
    negative — but the regressor is unconstrained and lands slightly below zero
    on the gentlest dishes. Clipping costs nothing (it improves MAE by ~0.02 on
    held-out folds) and avoids showing a physically meaningless number. The
    reported evaluation metrics do NOT include this clip, so they are marginally
    conservative rather than flattered by it.
    """
    import xgboost as xgb

    dmatrix = xgb.DMatrix(rows, feature_names=spec()["columns"])
    return np.clip(model().predict(dmatrix), 0.0, None)


# ---------------------------------------------------------------- public API

def predict_response(labs: dict, meal_id: str, *, with_curve: bool = True) -> dict:
    """Predict one person's glucose response to one standardized dish."""
    row, imputed = build_row(labs, meal_id)
    predicted = round(float(_predict_raw(row)[0]), 1)

    dish = dish_catalog().set_index("meal_id").loc[meal_id]
    observed_median = float(dish["observed_median_iauc"])

    out = {
        "meal_id": meal_id,
        "predicted_iauc": round(predicted, 1),
        "units": spec()["target_units"],
        "cohort_median_iauc": round(observed_median, 1),
        "vs_cohort": round(predicted - observed_median, 1),
        "dish": {
            "carbs_g": float(dish["carbs"]), "calories_kcal": float(dish["calories"]),
            "protein_g": float(dish["protein"]), "fat_g": float(dish["fat"]),
            "fiber_g": float(dish["fiber"]), "meal_type": str(dish["meal_type"]),
        },
        "imputed_fields": imputed,
        "model_version": manifest().get("git_sha", "unknown"),
    }
    if with_curve:
        curve = dish_curve(meal_id)
        scale = predicted / observed_median if observed_median > 0 else 1.0
        out["curve"] = [
            {"minute": int(m), "delta": round(float(d) * scale, 1)}
            for m, d in zip(curve["minute"], curve["delta"])
        ]
        out["curve_note"] = (
            "Shape is the cohort's median observed curve for this dish, scaled to "
            "the predicted iAUC. The model predicts the area, not the shape."
        )
    return out


def rank_meals(labs: dict, *, ascending: bool = True) -> pd.DataFrame:
    """Rank every standardized dish by this person's predicted response.

    This is the question the dataset can actually answer: given a lab panel,
    which of these meals is the gentlest for *this* person? Ascending order puts
    the best-tolerated dish first.
    """
    catalog = dish_catalog()
    rows, imputed = [], None
    for meal_id in catalog["meal_id"]:
        row, imputed = build_row(labs, meal_id)
        rows.append(row[0])
    predictions = _predict_raw(np.vstack(rows).astype(np.float32))

    out = catalog[["meal_id", "meal_type", "carbs", "calories",
                   "observed_median_iauc", "n_subjects"]].copy()
    out["observed_median_iauc"] = out["observed_median_iauc"].round(1)
    out["carbs"] = out["carbs"].astype(int)
    out["calories"] = out["calories"].astype(int)
    out["predicted_iauc"] = predictions.astype(float).round(1)
    out["vs_cohort"] = (out["predicted_iauc"] - out["observed_median_iauc"]).round(1)
    out = out.sort_values("predicted_iauc", ascending=ascending).reset_index(drop=True)
    out.attrs["imputed_fields"] = imputed
    return out


def manifest() -> dict:
    path = ARTIFACTS / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else {}


def required_labs() -> list[str]:
    """The lab fields the model uses. All are optional; medians fill the gaps."""
    return [c for c in spec()["lab_columns"] if c != "homa_ir"]


# ---------------------------------------------------------------------- CLI

def _demo() -> None:
    catalog = dish_catalog()
    print(f"loaded {len(catalog)} standardized dishes; "
          f"model expects {len(spec()['columns'])} features\n")
    print("lab fields used:", ", ".join(required_labs()), "\n")

    # Two ILLUSTRATIVE panels — invented, clinically plausible values, not copied
    # from any study participant. The repository does not redistribute the data,
    # and that includes not hardcoding real people's bloodwork into a demo.
    healthy = {"age": 30, "bmi": 22.0, "body_weight": 140, "height": 67,
               "a1c_pdl_lab": 5.2, "fasting_glu___pdl_lab": 88, "insulin": 4.0,
               "triglycerides": 70, "cholesterol": 175, "hdl": 65, "non_hdl": 110,
               "ldl_cal": 95, "vldl_cal": 14, "cho_hdl_ratio": 2.7}
    resistant = {"age": 55, "bmi": 31.0, "body_weight": 200, "height": 68,
                 "a1c_pdl_lab": 6.2, "fasting_glu___pdl_lab": 115, "insulin": 18.0,
                 "triglycerides": 190, "cholesterol": 210, "hdl": 38, "non_hdl": 172,
                 "ldl_cal": 130, "vldl_cal": 38, "cho_hdl_ratio": 5.5}

    print("(panels below are illustrative, not real participants)\n")
    for label, panel in [("metabolically healthy", healthy), ("insulin resistant", resistant)]:
        ranked = rank_meals(panel)
        homa = panel["fasting_glu___pdl_lab"] * panel["insulin"] / 405
        print(f"--- {label} (HOMA-IR {homa:.1f}) ---")
        print(ranked.head(4).to_string(index=False))
        print(f"    predicted range across dishes: "
              f"{ranked.predicted_iauc.min():.0f} to {ranked.predicted_iauc.max():.0f}\n")

    worst = rank_meals(resistant, ascending=False).iloc[0]
    detail = predict_response(resistant, worst["meal_id"])
    print(f"detail for the worst dish ({worst['meal_id']}):")
    for key in ("predicted_iauc", "cohort_median_iauc", "vs_cohort", "imputed_fields"):
        print(f"  {key:<22} {detail[key]}")
    print(f"  curve points           {len(detail['curve'])}")


if __name__ == "__main__":
    _demo()
