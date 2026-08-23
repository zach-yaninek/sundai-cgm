"""stub_server.py — the API with fixtures instead of a model.

    uv run --with fastapi --with uvicorn python contract/stub_server.py
    # or: pip install fastapi uvicorn && python contract/stub_server.py

Serves every endpoint in `openapi.json` with plausible, *input-dependent* fixtures
and no model, no data files and no dependency on anything under `artifacts/`.

It exists so the frontend can be built and finished before the real backend lands.
`serve.py` will implement the identical shapes; `test_contract.py` checks both
against the same spec, so a screen built here keeps working after the swap.

The fixtures respond to the input on purpose — a higher carbohydrate load and a
worse lab panel produce a higher probability, a wider confidence band and different
edits — so the frontend can exercise every visual state without waiting for a model.
Nothing here is a prediction. Every number is fabricated.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

HERE = Path(__file__).parent
SPEC = json.loads((HERE / "openapi.json").read_text())

app = FastAPI(
    title="sundai-cgm meal risk API (STUB)",
    version=SPEC["info"]["version"],
    description="Fixture server. No model. See contract/openapi.json.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STUB_VERSION = "stub-0000000"

# Ordered by measured importance in the real model, so the onboarding form is
# already in the right order when the real backend replaces this.
FIELDS = [
    ("fasting_glu___pdl_lab", "Fasting glucose", "mg/dL", 40, 400, 97.0, 1),
    ("insulin", "Fasting insulin", "uIU/mL", 0, 300, 9.4, 2),
    ("a1c_pdl_lab", "HbA1c", "%", 3, 15, 5.5, 3),
    ("vldl_cal", "VLDL", "mg/dL", 1, 200, 18.0, 4),
    ("triglycerides", "Triglycerides", "mg/dL", 10, 2000, 92.0, 5),
    ("cholesterol", "Total cholesterol", "mg/dL", 50, 500, 180.0, 6),
    ("bmi", "BMI", "kg/m2", 12, 70, 27.4, 7),
    ("hdl", "HDL", "mg/dL", 10, 150, 55.0, 8),
    ("ldl_cal", "LDL", "mg/dL", 10, 400, 105.0, 9),
    ("cho_hdl_ratio", "Cholesterol / HDL", "ratio", 0.5, 20, 3.3, 10),
    ("non_hdl", "Non-HDL cholesterol", "mg/dL", 10, 400, 125.0, 11),
    ("age", "Age", "years", 18, 100, 39.0, 12),
    ("body_weight", "Weight", "lb", 60, 600, 172.0, 13),
    ("height", "Height", "in", 48, 90, 66.0, 14),
]

DISCLAIMER = (
    "This is a research demo, not a medical device. Estimates come from a model "
    "fitted to 45 adults in a single study and have not been clinically validated. "
    "Do not use it to decide an insulin dose or to change how you manage a diagnosed "
    "condition. Talk to a clinician about your own numbers."
)

EXCLUSIONS = [
    "You have type 1 diabetes",
    "You would use this to inform an insulin dose",
    "You take glucose-lowering medication",
    "You are pregnant",
    "You are under 18",
]


MEAL_RANGES = {"carbs": (0, 800), "protein": (0, 400), "fat": (0, 400),
               "fiber": (0, 100), "calories": (0, 4000)}


def _validate(labs: dict, meal: dict, pre: float | None) -> str | None:
    """Same refusals as the real backend, so the frontend builds the error path.

    A stub that accepts everything teaches the UI that bad input never fails,
    and the validation states go unbuilt until the real backend starts rejecting
    things in front of an audience.
    """
    ranges = {f[0]: (f[3], f[4]) for f in FIELDS}
    for name, value in (labs or {}).items():
        if name not in ranges:
            return f"unknown lab field {name!r}"
        lo, hi = ranges[name]
        if value is not None and not (lo <= float(value) <= hi):
            return f"{name} is {value}, outside the plausible range {lo}-{hi}"
    for name, (lo, hi) in MEAL_RANGES.items():
        value = (meal or {}).get(name)
        if value is not None and not (lo <= float(value) <= hi):
            return f"meal.{name} is {value}, outside the plausible range {lo}-{hi}"
    if pre is not None and not (40 <= float(pre) <= 400):
        return f"pre_meal_glucose is {pre}, outside the plausible range 40-400"
    return None


def _meta() -> dict[str, Any]:
    return {
        "model_version": STUB_VERSION,
        "cohort": {
            "n_subjects": 45,
            "n_meals": 1382,
            "age_range": "24-59",
            "source": "CGMacros (PhysioNet 2025)",
            "citation": "Gutierrez-Osuna et al., doi:10.13026/3z8q-x658. CC BY-NC-SA 4.0.",
        },
        "thresholds": {"exceed_mgdl": 140, "cohort_exceed_rate": 0.458},
        "performance": {
            "auc_with_glucose": 0.8854,
            "auc_without_glucose": 0.8358,
            "mae_iauc": 28.172,
            "learning_curve": [
                {"meals_logged": 0, "mae": 28.662},
                {"meals_logged": 1, "mae": 26.687},
                {"meals_logged": 2, "mae": 25.887},
                {"meals_logged": 3, "mae": 25.668},
                {"meals_logged": 5, "mae": 24.443},
                {"meals_logged": 10, "mae": 23.7},
                {"meals_logged": 15, "mae": 23.436},
            ],
        },
        "disclaimer": {"id": "v1", "text": DISCLAIMER, "must_accept": True},
        "exclusions": EXCLUSIONS,
    }


def _jitter(payload: dict, lo: float, hi: float) -> float:
    """Deterministic pseudo-noise, so identical requests give identical answers."""
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode())
    return lo + (int(digest.hexdigest()[:8], 16) / 0xFFFFFFFF) * (hi - lo)


def _risk(labs: dict, meal: dict, pre: float | None) -> tuple[float, float]:
    """A crude stand-in that moves in the directions the real model moves."""
    carbs = float(meal.get("carbs") or 0)
    fiber = float(meal.get("fiber") or 0)
    eaten = 1.0
    a1c = float(labs.get("a1c_pdl_lab") or 5.5)
    glu = float(labs.get("fasting_glu___pdl_lab") or 97)
    ins = float(labs.get("insulin") or 9.4)
    homa = glu * ins / 405.0

    score = (
        0.016 * carbs * eaten
        - 0.030 * fiber
        + 0.55 * (a1c - 5.5)
        + 0.14 * (homa - 2.2)
        + 0.012 * ((pre if pre else glu) - 97)
        - 1.15
    )
    probability = 1 / (1 + pow(2.718281828, -score))
    baseline = pre if pre else glu
    peak = baseline + 18 + 0.55 * carbs * eaten - 1.1 * fiber + 9 * (a1c - 5.5)
    return round(min(max(probability, 0.01), 0.99), 3), round(peak, 1)


def _curve(peak_delta: float) -> list[dict]:
    """A plausible postprandial shape: rise to ~45 min, decay by 120."""
    shape = [0.00, 0.18, 0.45, 0.72, 0.92, 1.00, 0.97, 0.88, 0.76, 0.63,
             0.52, 0.42, 0.34, 0.27, 0.22, 0.18, 0.14, 0.11, 0.09, 0.07,
             0.05, 0.04, 0.03, 0.02, 0.01]
    return [{"minute": i * 5, "delta": round(peak_delta * s, 1)} for i, s in enumerate(shape)]


def _confidence(labs: dict, pre: float | None) -> dict:
    supplied = {k for k, v in labs.items() if v is not None}
    imputed = [f[0] for f in FIELDS if f[0] not in supplied]
    if len(imputed) >= 8 or (len(imputed) >= 5 and pre is None):
        band = "wide"
    elif imputed or pre is None:
        band = "moderate"
    else:
        band = "narrow"
    return {"band": band, "imputed_fields": imputed, "used_pre_meal_glucose": pre is not None}


def _personalization(history: list) -> dict:
    k = len(history)
    shrinkage = k / (k + 5) if k else 0.0
    curve = {p["meals_logged"]: p["mae"] for p in _meta()["performance"]["learning_curve"]}
    nearest = min(curve, key=lambda m: abs(m - k))
    return {
        "meals_logged": k,
        "offset_applied": round(shrinkage * 12.0, 1),   # fixture direction only
        "shrinkage": round(shrinkage, 3),
        "expected_mae": curve[nearest],
    }


@app.get("/api/meta")
def meta():
    return _meta()


@app.get("/api/fields")
def fields():
    return {
        "fields": [
            {"name": n, "label": lbl, "unit": u, "optional": True,
             "min": lo, "max": hi, "cohort_median": med, "importance_rank": rank}
            for n, lbl, u, lo, hi, med, rank in FIELDS
        ],
        "derived": {"homa_ir": "fasting glucose * insulin / 405"},
    }


@app.post("/api/assess")
def assess(body: dict):
    labs = body.get("labs") or {}
    meal = body.get("meal") or {}
    pre = body.get("pre_meal_glucose")
    history = body.get("history") or []

    if "carbs" not in meal or "meal_type" not in meal:
        return JSONResponse(
            status_code=422,
            content={"error": "invalid_meal",
                     "detail": "carbs, calories and meal_type are required",
                     "field": "meal"},
        )
    problem = _validate(labs, meal, pre)
    if problem:
        return JSONResponse(status_code=422,
                            content={"error": "invalid_request",
                                     "detail": problem, "field": None})

    probability, peak = _risk(labs, meal, pre)
    personal = _personalization(history)
    peak += personal["offset_applied"] * 0.4
    baseline = pre if pre else float(labs.get("fasting_glu___pdl_lab") or 97)
    iauc = max(0.0, round((peak - baseline) * 1.35 + _jitter(body, -4, 4), 1))

    return {
        "exceeds_140": {
            "threshold_mgdl": 140,
            "probability": probability,
            "cohort_percentile": int(round(probability * 96)),
            "operating_point": "recall_weighted",
        },
        "predicted_peak_mgdl": round(peak, 1),
        "predicted_iauc": iauc,
        "curve": _curve(max(0.0, peak - baseline)),
        "curve_note": "Cohort median shape for meals like this, scaled to the prediction.",
        "personalization": personal,
        "confidence": _confidence(labs, pre),
        "model_version": STUB_VERSION,
        "disclaimer_id": "v1",
    }


@app.post("/api/alternatives")
def alternatives(body: dict):
    labs = body.get("labs") or {}
    meal = dict(body.get("meal") or {})
    pre = body.get("pre_meal_glucose")
    history = body.get("history") or []
    target = float(body.get("target_probability", 0.4))

    problem = _validate(labs, meal, pre)
    if problem:
        return JSONResponse(status_code=422,
                            content={"error": "invalid_request",
                                     "detail": problem, "field": None})

    base_p, _ = _risk(labs, meal, pre)
    carbs = float(meal.get("carbs") or 0)

    candidates = [
        ("About a quarter less carbohydrate", {"carbs": -round(carbs * 0.25, 1)}),
        ("Add 10 g of fiber", {"fiber": 10}),
        ("About a third less carbohydrate", {"carbs": -round(carbs * 0.33, 1)}),
        ("Half the carbohydrate", {"carbs": -round(carbs * 0.5, 1)}),
    ]

    edits = []
    known = ("carbs", "protein", "fat", "fiber", "calories", "meal_type", "label")
    for description, changes in candidates:
        variant = {k: v for k, v in meal.items() if k in known}
        for key, delta in changes.items():
            variant[key] = max(0.0, float(variant.get(key) or 0) + delta)
        p, peak = _risk(labs, variant, pre)
        if p < base_p:
            edits.append({
                "description": description,
                "changes": changes,
                "resulting_meal": variant,
                "probability": p,
                "delta_probability": round(p - base_p, 3),
                "predicted_peak_mgdl": peak,
            })
        if p <= target:
            break

    past = [
        {"meal": h["meal"], "observed_peak": h.get("observed_peak"),
         "observed_iauc": h.get("observed_iauc"), "logged_at": h.get("logged_at")}
        for h in history
        if h.get("meal", {}).get("meal_type") == meal.get("meal_type")
        and (h.get("observed_peak") or 999) < 140
    ][:3]

    note = ("This meal is already in the lower range for you."
            if not edits else
            "Changes this model predicts would lower your response. Not dietary advice.")

    return {
        "original": {
            "threshold_mgdl": 140, "probability": base_p,
            "cohort_percentile": int(round(base_p * 96)),
            "operating_point": "recall_weighted",
        },
        "edits": edits,
        "from_your_history": past,
        "note": note,
        "model_version": STUB_VERSION,
    }


if __name__ == "__main__":
    import uvicorn

    print("STUB server — fixtures only, no model. Contract: contract/openapi.json")
    uvicorn.run(app, host="127.0.0.1", port=8000)
