"""explain.py — say *why*, grounded in what the model actually did.

The prediction is a number. This turns it into a sentence, using real
per-prediction attributions from XGBoost rather than letting a language model
infer what mattered. Claude's job here is narrow: verbalise the drivers we hand
it. It does not decide which features were important, and it never sees the
person's lab values.

    import explain
    explain.explain(labs, meal, pre_meal_glucose, assessment)

Three properties make this safe enough to put in front of someone:

**The upstream call never receives raw patient data.** It gets feature *names*
and *contribution magnitudes* — "a1c contributed +0.73" — plus the numbers the UI
is already showing. Not "your HbA1c is 6.2". That keeps health values out of a
third-party request, and it removes the model's ability to invent one.

**Nothing it says can contain a number we did not give it.** Every numeric token
in the response is checked against the payload. A single invented figure and the
whole response is discarded.

**It degrades instead of breaking.** With no API key, a failed call, or a
response that fails validation, a deterministic template built from the same
drivers is returned. The app never depends on a network call to explain itself.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

import numpy as np

import risk

MODEL = "claude-opus-5"
MAX_DRIVERS = 4

# Phrases that would undo the product's copy rules in one sentence. Checked
# mechanically because "the prompt says not to" is not an assurance.
BANNED = (
    "dangerous", "danger", "diagnos", "you should eat", "you must",
    "treat", "cure", "prescri", "medical advice", "safe to eat",
)

EXPLANATION_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {
            "type": "string",
            "description": "One sentence, under 25 words, on what drove this estimate.",
        },
        "drivers": {
            "type": "array",
            "items": {"type": "string"},
            "description": "One short clause per driver, in the order supplied.",
        },
        "caveat": {
            "type": "string",
            "description": "One sentence naming the biggest limitation of this estimate.",
        },
    },
    "required": ["headline", "drivers", "caveat"],
    "additionalProperties": False,
}

SYSTEM = """You explain the output of a glucose-response model to the person it \
was run for. You are given the model's own feature attributions and the figures \
already displayed on screen. Your only job is to put those into plain language.

Hard rules:
- Use only the numbers supplied. Never introduce a figure that is not in the input.
- Never say "dangerous", never diagnose, never tell the person what to eat.
- Say "this model predicts", not "you will".
- The model was fitted to 45 people in one study. Say so in the caveat when the
  confidence band is wide or fields were imputed.
- Plain clinical English. No hedging stacks, no exclamation marks, no emoji."""

# Human labels for feature names. Keeps the model from having to guess what
# `cho_hdl_ratio` or `a1c_pdl_lab` mean, and keeps the wording consistent.
LABELS = {
    "carbs": "carbohydrate in the meal", "protein": "protein in the meal",
    "fat": "fat in the meal", "fiber": "fibre in the meal",
    "calories": "calories in the meal", "pre_meal_glucose": "pre-meal glucose reading",
    "a1c_pdl_lab": "HbA1c", "insulin": "fasting insulin", "homa_ir": "insulin resistance (HOMA-IR)",
    "fasting_glu___pdl_lab": "fasting glucose", "triglycerides": "triglycerides",
    "cholesterol": "total cholesterol", "hdl": "HDL", "non_hdl": "non-HDL cholesterol",
    "ldl_cal": "LDL", "vldl_cal": "VLDL", "cho_hdl_ratio": "cholesterol-to-HDL ratio",
    "bmi": "BMI", "age": "age", "body_weight": "body weight", "height": "height",
    "mt_breakfast": "the meal being breakfast", "mt_lunch": "the meal being lunch",
    "mt_dinner": "the meal being dinner", "mt_snack": "the meal being a snack",
}

_CACHE: dict[str, dict] = {}


# ------------------------------------------------------------------ drivers

def drivers(labs: dict, meal: dict, pre_meal_glucose: float | None,
            *, top: int = MAX_DRIVERS) -> list[dict]:
    """Per-prediction attributions for the risk flag, strongest first.

    These are the model's own SHAP contributions, not a guess about which inputs
    matter in general. ``value`` is deliberately absent — only the feature's name
    and how much it pushed this particular prediction.
    """
    import xgboost as xgb

    variant = risk.variant_for(pre_meal_glucose)
    columns = risk.spec()["variants"][variant]["columns"]
    row, _ = risk.build_row(labs, meal, pre_meal_glucose)

    booster = risk._booster(variant, "exceeds")
    matrix = xgb.DMatrix(row, feature_names=columns)
    contribs = booster.predict(matrix, pred_contribs=True)[0]

    ranked = sorted(
        ((columns[i], float(contribs[i])) for i in range(len(columns))),
        key=lambda kv: -abs(kv[1]),
    )
    out = []
    for name, contribution in ranked[:top]:
        if abs(contribution) < 0.01:
            continue
        out.append({
            "feature": name,
            "label": LABELS.get(name, name),
            "direction": "raises" if contribution > 0 else "lowers",
            "contribution": round(abs(contribution), 3),
        })
    return out


def build_payload(assessment: dict, attributions: list[dict],
                  *, meal_type: str | None = None) -> dict:
    """Exactly what is sent upstream. No lab values, no meal macros.

    Everything here is either a figure already on the user's screen or an
    attribution magnitude. If you add a field, check it against that rule first.
    """
    return {
        "probability_exceeds_threshold": assessment["probability"],
        "threshold_mgdl": assessment["threshold_mgdl"],
        "cohort_percentile": assessment["cohort_percentile"],
        "predicted_peak_mgdl": assessment["predicted_peak_mgdl"],
        "confidence_band": assessment["confidence_band"],
        "n_imputed_fields": len(assessment.get("imputed_fields", [])),
        "used_pre_meal_glucose": assessment["variant"] == "with_glucose",
        "meal_type": meal_type,
        "cohort_size": 45,
        "drivers": attributions,
    }


# ----------------------------------------------------------------- guardrails

def _numbers_in(text: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?", text)


def _allowed_numbers(payload: dict) -> set[str]:
    """Every rendering of a supplied number that we will accept in prose."""
    allowed: set[str] = set()

    def add(value: Any) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return
        allowed.update({
            f"{value:g}", f"{round(value)}", f"{value:.1f}", f"{value:.2f}",
            f"{round(value * 100)}", f"{value * 100:.1f}",
        })

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        else:
            add(node)

    walk(payload)
    # Small integers are how anything counts its own list ("all three drivers").
    allowed.update(str(i) for i in range(0, 11))
    return allowed


def validate(explanation: dict, payload: dict) -> tuple[bool, str]:
    """Reject anything that invents a figure or breaks a copy rule.

    Returns ``(ok, reason)``. The caller falls back to the template on failure —
    a wrong explanation next to a correct number is worse than a plain one.
    """
    if not isinstance(explanation, dict):
        return False, "not an object"
    for key in ("headline", "drivers", "caveat"):
        if key not in explanation:
            return False, f"missing {key}"

    text = " ".join([
        str(explanation["headline"]),
        " ".join(str(d) for d in explanation["drivers"]),
        str(explanation["caveat"]),
    ])

    lowered = text.lower()
    for phrase in BANNED:
        if phrase in lowered:
            return False, f"banned phrase: {phrase!r}"

    allowed = _allowed_numbers(payload)
    for number in _numbers_in(text):
        if number not in allowed:
            return False, f"invented number: {number}"

    if len(str(explanation["headline"]).split()) > 40:
        return False, "headline too long"
    return True, "ok"


# ------------------------------------------------------------------ fallback

def template_explanation(payload: dict) -> dict:
    """Deterministic explanation from the same drivers. No network, no key.

    This is the default path until an API key exists, so it has to read well on
    its own rather than being an apology for a missing feature.
    """
    pct = round(payload["probability_exceeds_threshold"] * 100)
    threshold = payload["threshold_mgdl"]
    attributions = payload.get("drivers") or []

    if attributions:
        lead = attributions[0]
        headline = (
            f"This model puts the chance of going above {threshold:g} mg/dL at "
            f"{pct}%, driven most by {lead['label']}."
        )
    else:
        headline = (f"This model puts the chance of going above {threshold:g} mg/dL "
                    f"at {pct}%.")

    lines = [f"{d['label']} {d['direction']} the estimate" for d in attributions]

    band = payload.get("confidence_band")
    missing = payload.get("n_imputed_fields", 0)
    caveat = f"Fitted to {payload.get('cohort_size', 45)} people in one study"
    if band == "wide":
        caveat += f", and {missing} inputs were filled from cohort medians here"
    elif not payload.get("used_pre_meal_glucose"):
        caveat += ", and no pre-meal glucose reading was supplied"
    caveat += "."

    return {"headline": headline, "drivers": lines, "caveat": caveat,
            "source": "template"}


# --------------------------------------------------------------------- main

def _cache_key(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def available() -> bool:
    """Whether a live explanation is possible at all."""
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def explain(labs: dict, meal: dict, pre_meal_glucose: float | None,
            assessment: dict, *, client: Any = None) -> dict:
    """Narrate one assessment. Always returns something usable.

    ``client`` is injectable so the guardrails can be tested without credentials
    — which matters, because the guardrails are the part worth testing.
    """
    attributions = drivers(labs, meal, pre_meal_glucose)
    payload = build_payload(assessment, attributions,
                            meal_type=meal.get("meal_type"))

    key = _cache_key(payload)
    if key in _CACHE:
        return _CACHE[key]

    result = _live(payload, client) if (client or available()) else None
    if result is None:
        result = template_explanation(payload)

    result["drivers_used"] = attributions
    _CACHE[key] = result
    return result


def _live(payload: dict, client: Any = None) -> dict | None:
    """One call to Claude. Returns None on any failure — the caller falls back."""
    if client is None:
        # The SDK is only needed to construct a default client. Importing it
        # unconditionally would make an injected client unusable, which is
        # exactly how the guardrails get tested without credentials.
        try:
            import anthropic
        except ImportError:
            return None
        try:
            client = anthropic.Anthropic()
        except Exception:
            return None

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=[{"type": "text", "text": SYSTEM,
                     "cache_control": {"type": "ephemeral"}}],
            output_config={
                "effort": "low",  # verbalising supplied facts, not reasoning
                "format": {"type": "json_schema", "schema": EXPLANATION_SCHEMA},
            },
            messages=[{"role": "user", "content": json.dumps(payload, indent=2)}],
        )
    except Exception:
        # Any upstream failure is a fallback, never an error the user sees.
        return None

    if getattr(response, "stop_reason", None) == "refusal":
        return None

    try:
        text = next(b.text for b in response.content if b.type == "text")
        parsed = json.loads(text)
    except (StopIteration, AttributeError, json.JSONDecodeError):
        return None

    ok, reason = validate(parsed, payload)
    if not ok:
        parsed = template_explanation(payload)
        parsed["rejected_reason"] = reason
        return parsed

    parsed["source"] = "claude"
    return parsed
