"""serve.py — the real backend, behind the contract the stub already implements.

    uv run --with fastapi --with uvicorn --with numpy --with xgboost python serve.py
    # or: pip install fastapi uvicorn && python serve.py

Same six endpoints and the same response shapes as `contract/stub_server.py`, so
the frontend switches over by changing nothing at all. `test_contract.py --real`
validates both against `contract/openapi.json`.

Stateless on purpose. The lab panel and the meal history arrive with each request
and are never written anywhere — that is the app's privacy story, and it is only
true as long as this file stays free of storage.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import explain
import recommend
import risk
# Only the shrinkage maths, not personalize.py — that module imports pandas,
# xgboost and the training code to *measure* the learning curve, none of which
# the serving container should have to carry.
import shrinkage as shrink
from sundai_cgm import value_of_information

HERE = Path(__file__).parent
ARTIFACTS = HERE / "artifacts"

app = FastAPI(title="sundai-cgm meal risk API", version="1.0.0")
# Deployed, the frontend is on a different origin, so the allowed list has to be
# configurable. Defaults to the local dev server so nothing changes locally.
ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get(
        "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

DISCLAIMER = (
    "This is a research demo, not a medical device. Estimates come from a model "
    "fitted to 45 adults in a single study and have not been clinically validated. "
    "Do not use it to decide an insulin dose or to change how you manage a "
    "diagnosed condition. Talk to a clinician about your own numbers."
)

EXCLUSIONS = [
    "You have type 1 diabetes",
    "You would use this to inform an insulin dose",
    "You take glucose-lowering medication",
    "You are pregnant",
    "You are under 18",
]

# Ranges for the onboarding form. Wide enough to admit real outliers, tight
# enough that a typo'd 660 for 66 is refused rather than scored.
FIELD_META = {
    "fasting_glu___pdl_lab": ("Fasting glucose", "mg/dL", 40, 400),
    "insulin": ("Fasting insulin", "uIU/mL", 0, 300),
    "a1c_pdl_lab": ("HbA1c", "%", 3, 15),
    "vldl_cal": ("VLDL", "mg/dL", 1, 200),
    "triglycerides": ("Triglycerides", "mg/dL", 10, 2000),
    "cholesterol": ("Total cholesterol", "mg/dL", 50, 500),
    "bmi": ("BMI", "kg/m2", 12, 70),
    "hdl": ("HDL", "mg/dL", 10, 150),
    "ldl_cal": ("LDL", "mg/dL", 10, 400),
    "cho_hdl_ratio": ("Cholesterol / HDL", "ratio", 0.5, 20),
    "non_hdl": ("Non-HDL cholesterol", "mg/dL", 10, 400),
    "age": ("Age", "years", 18, 100),
    "body_weight": ("Weight", "lb", 60, 600),
    "height": ("Height", "in", 48, 90),
}

MEAL_RANGES = {
    "carbs": (0, 800), "protein": (0, 400), "fat": (0, 400),
    "fiber": (0, 100), "calories": (0, 4000),
}


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                              capture_output=True, text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


MODEL_VERSION = _git_sha()


def _learning_curve() -> list[dict]:
    path = ARTIFACTS / "learning_curve.json"
    if not path.exists():
        return []
    return [{"meals_logged": p["meals_logged"], "mae": p["mae"]}
            for p in json.loads(path.read_text())["points"]]


def _error(detail: str, field: str | None = None, code: str = "invalid_request"):
    return JSONResponse(status_code=422,
                        content={"error": code, "detail": detail, "field": field})


def _validate(labs: dict, meal: dict, pre: float | None) -> str | None:
    """Refuse implausible input rather than scoring it.

    A model will return a confident number for 660 g of carbohydrate. Catching
    the typo here is the difference between a wrong answer and no answer.
    """
    for name, value in (labs or {}).items():
        meta = FIELD_META.get(name)
        if meta is None:
            return f"unknown lab field {name!r}"
        _, _, lo, hi = meta
        if value is not None and not (lo <= float(value) <= hi):
            return f"{name} is {value}, outside the plausible range {lo}-{hi}"
    for name, (lo, hi) in MEAL_RANGES.items():
        value = (meal or {}).get(name)
        if value is not None and not (lo <= float(value) <= hi):
            return f"meal.{name} is {value}, outside the plausible range {lo}-{hi}"
    if pre is not None and not (40 <= float(pre) <= 400):
        return f"pre_meal_glucose is {pre}, outside the plausible range 40-400"
    return None


def _curve(peak_delta: float) -> list[dict]:
    """Cohort median postprandial shape, scaled to this prediction.

    The model predicts the area and the peak, not the trajectory. This is an
    observed average shape, and the response says so.
    """
    shape = [0.00, 0.18, 0.45, 0.72, 0.92, 1.00, 0.97, 0.88, 0.76, 0.63,
             0.52, 0.42, 0.34, 0.27, 0.22, 0.18, 0.14, 0.11, 0.09, 0.07,
             0.05, 0.04, 0.03, 0.02, 0.01]
    return [{"minute": i * 5, "delta": round(max(0.0, peak_delta) * s, 1)}
            for i, s in enumerate(shape)]


def _assemble(scored: dict, labs: dict, history: list, offset: float,
              k: int, pre: float | None) -> dict:
    baseline = pre if pre is not None else labs.get("fasting_glu___pdl_lab")
    peak_delta = scored["predicted_peak_mgdl"] - float(baseline or 97)
    curve = _learning_curve()
    expected = min(curve, key=lambda p: abs(p["meals_logged"] - k))["mae"] if curve else None

    return {
        "exceeds_140": {
            "threshold_mgdl": scored["threshold_mgdl"],
            "probability": scored["probability"],
            "cohort_percentile": scored["cohort_percentile"],
            "operating_point": "recall_weighted",
        },
        "predicted_peak_mgdl": scored["predicted_peak_mgdl"],
        "predicted_iauc": scored["predicted_iauc"],
        "curve": _curve(peak_delta),
        "curve_note": ("Cohort median shape for a meal like this, scaled to the "
                       "prediction. The model predicts the area and the peak, not "
                       "the trajectory."),
        "personalization": {
            "meals_logged": k,
            "offset_applied": round(offset, 1),
            "shrinkage": round(shrink.shrinkage(k), 3),
            **({"expected_mae": expected} if expected is not None else {}),
        },
        "confidence": {
            "band": scored["confidence_band"],
            "imputed_fields": scored["imputed_fields"],
            "used_pre_meal_glucose": pre is not None,
        },
        "model_version": MODEL_VERSION,
        "disclaimer_id": "v1",
    }


@app.get("/api/meta")
def meta():
    perf = risk.performance()
    return {
        "model_version": MODEL_VERSION,
        "cohort": {
            "n_subjects": 45,
            "n_meals": 1382,
            "age_range": "24-59",
            "source": "CGMacros (PhysioNet 2025)",
            "citation": "Gutierrez-Osuna et al., doi:10.13026/3z8q-x658. CC BY-NC-SA 4.0.",
        },
        "thresholds": {
            "exceed_mgdl": risk.spec()["threshold_mgdl"],
            "cohort_exceed_rate": 0.458,
        },
        "performance": {
            "auc_with_glucose": perf["with_glucose"]["auc"],
            "auc_without_glucose": perf["no_glucose"]["auc"],
            "mae_iauc": perf["with_glucose"]["mae_iauc"],
            "learning_curve": _learning_curve(),
        },
        "disclaimer": {"id": "v1", "text": DISCLAIMER, "must_accept": True},
        "exclusions": EXCLUSIONS,
    }


@app.get("/api/fields")
def fields():
    order = list(FIELD_META)
    return {
        "fields": [
            {
                "name": name,
                "label": FIELD_META[name][0],
                "unit": FIELD_META[name][1],
                "optional": True,
                "min": FIELD_META[name][2],
                "max": FIELD_META[name][3],
                "cohort_median": round(
                    risk.spec()["variants"]["no_glucose"]["impute_medians"][name], 2),
                "importance_rank": order.index(name) + 1,
            }
            for name in order if name in risk.lab_fields()
        ],
        "derived": {"homa_ir": "fasting glucose * insulin / 405"},
    }


@app.post("/api/assess")
def assess(body: dict):
    labs = body.get("labs") or {}
    meal = body.get("meal") or {}
    pre = body.get("pre_meal_glucose")
    history = body.get("history") or []

    problem = _validate(labs, meal, pre)
    if problem:
        return _error(problem)

    try:
        offset, k = recommend.personal_offset(labs, history, pre)
        scored = risk.score(labs, meal, pre, offset=offset)
    except ValueError as exc:
        return _error(str(exc), field="meal", code="invalid_meal")
    except risk.ArtifactError as exc:
        return JSONResponse(status_code=503,
                            content={"error": "artifacts_unavailable",
                                     "detail": str(exc), "field": None})
    return _assemble(scored, labs, history, offset, k, pre)


@app.post("/api/alternatives")
def alternatives(body: dict):
    labs = body.get("labs") or {}
    meal = body.get("meal") or {}
    pre = body.get("pre_meal_glucose")
    history = body.get("history") or []
    target = float(body.get("target_probability", recommend.DEFAULT_TARGET))

    problem = _validate(labs, meal, pre)
    if problem:
        return _error(problem)

    try:
        offset, _ = recommend.personal_offset(labs, history, pre)
        result = recommend.suggest(labs, meal, pre, offset=offset,
                                   target_probability=target)
    except ValueError as exc:
        return _error(str(exc), field="meal", code="invalid_meal")
    except risk.ArtifactError as exc:
        return JSONResponse(status_code=503,
                            content={"error": "artifacts_unavailable",
                                     "detail": str(exc), "field": None})

    base = result["original"]
    return {
        "original": {
            "threshold_mgdl": base["threshold_mgdl"],
            "probability": base["probability"],
            "cohort_percentile": base["cohort_percentile"],
            "operating_point": "recall_weighted",
        },
        "edits": [
            {
                "description": e["description"],
                "changes": e["changes"],
                "resulting_meal": e["resulting_meal"],
                "probability": e["probability"],
                "delta_probability": e["delta_probability"],
                "predicted_peak_mgdl": e["predicted_peak_mgdl"],
            }
            for e in result["edits"]
        ],
        "from_your_history": recommend.from_history(history, meal.get("meal_type")),
        "note": result["note"],
        "model_version": MODEL_VERSION,
    }


@app.post("/api/lab-value")
def lab_value(body: dict):
    """Would drawing this person's bloods sharpen what we can tell them?"""
    labs = body.get("labs") or {}
    pre = body.get("pre_meal_glucose")

    problem = _validate(labs, {}, pre)
    if problem:
        return _error(problem)

    assessment = value_of_information(labs, pre)
    return {**assessment, "model_version": MODEL_VERSION}


@app.post("/api/explain")
def explain_assessment(body: dict):
    """Narrate one assessment. Never fails — falls back to a template."""
    labs = body.get("labs") or {}
    meal = body.get("meal") or {}
    pre = body.get("pre_meal_glucose")
    history = body.get("history") or []

    problem = _validate(labs, meal, pre)
    if problem:
        return _error(problem)

    try:
        offset, _ = recommend.personal_offset(labs, history, pre)
        assessment = risk.score(labs, meal, pre, offset=offset)
        result = explain.explain(labs, meal, pre, assessment)
    except ValueError as exc:
        return _error(str(exc), field="meal", code="invalid_meal")
    except risk.ArtifactError as exc:
        return JSONResponse(status_code=503,
                            content={"error": "artifacts_unavailable",
                                     "detail": str(exc), "field": None})
    return {**result, "model_version": MODEL_VERSION}


if __name__ == "__main__":
    import uvicorn

    print(f"real backend — model {MODEL_VERSION}, artifacts from {ARTIFACTS}")
    uvicorn.run(app, host="127.0.0.1", port=8000)
