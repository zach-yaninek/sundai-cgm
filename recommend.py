"""recommend.py — what to change about this meal, for this person.

Given a meal the user entered, search interpretable single-axis edits and return
the **smallest change that gets the predicted probability below their band**.

Two deliberate design choices:

**Edits, not a catalog.** There is no database of alternative meals to pick from.
The 503 free-living meals in the study are CC BY-NC-SA data this project does not
redistribute, and "the same dinner with a third less rice" is more actionable than
"study meal #417" regardless. Everything here is a modification of the user's own
meal or a meal from their own history.

**Edits stay physiologically coherent.** Removing 20 g of carbohydrate also
removes ~80 kcal. Without that the model is asked to score an impossible meal —
less carbohydrate, identical calories — and it will happily return a number for it.
"""
from __future__ import annotations

import numpy as np

import risk

# kcal per gram, for keeping an edited meal internally consistent.
KCAL = {"carbs": 4.0, "protein": 4.0, "fat": 9.0, "fiber": 2.0}

# The only fields a returned meal may carry. A caller may send extras (an old
# client still posting `amount_consumed`, say); echoing them back would put a
# field in the response that the API contract forbids, so they are dropped here
# rather than trusted through.
MEAL_FIELDS = ("carbs", "protein", "fat", "fiber", "calories", "meal_type", "label")

# Ordered by how much they ask of the person. The first edit that reaches the
# target wins, so this order IS the "smallest effective change" rule.
EDIT_LADDER = [
    ("Add 10 g of fiber", {"fiber": +10}),
    ("A quarter less carbohydrate", {"carbs": -0.25}),
    ("Add 15 g of protein", {"protein": +15}),
    ("A third less carbohydrate", {"carbs": -0.33}),
    ("A quarter less of everything", {"carbs": -0.25, "protein": -0.25,
                                      "fat": -0.25, "fiber": -0.25}),
    ("Half the carbohydrate", {"carbs": -0.50}),
    ("Half the carbohydrate and 10 g more fiber", {"carbs": -0.50, "fiber": +10}),
    ("Half of everything", {"carbs": -0.50, "protein": -0.50,
                            "fat": -0.50, "fiber": -0.50}),
]

# Note on what is NOT here. "Eat three quarters of the portion" was an obvious
# candidate and is gone: it relied on `amount_consumed`, whose published values
# mix percentages, counts and gram-like numbers up to 900, so the model has no
# coherent portion axis to reason along. Scaling the macros down is the honest
# way to express the same idea. Fiber support is also thin -- corr(fiber, iauc)
# is only -0.05 -- but the search below drops any edit that does not actually
# lower this person's prediction, so a weak axis simply never gets surfaced.

DEFAULT_TARGET = 0.4


def apply_edit(meal: dict, changes: dict) -> tuple[dict, dict]:
    """Apply one edit, keeping calories consistent. Returns (meal, absolute deltas).

    Fractional values are proportions of the current amount; whole numbers are
    absolute grams. Calories follow the macros so the edited meal stays possible.
    """
    out = {k: v for k, v in meal.items() if k in MEAL_FIELDS}
    applied: dict[str, float] = {}
    calorie_delta = 0.0

    for field, change in changes.items():
        current = float(out.get(field) or 0.0)
        delta = current * change if abs(change) < 1.0 else float(change)
        new = max(0.0, current + delta)
        actual = new - current
        if actual == 0.0:
            continue
        applied[field] = round(actual, 1)
        out[field] = round(new, 1)
        calorie_delta += actual * KCAL.get(field, 0.0)

    if calorie_delta and out.get("calories") is not None:
        out["calories"] = round(max(0.0, float(out["calories"]) + calorie_delta), 1)
        applied["calories"] = round(calorie_delta, 1)

    return out, applied


def suggest(labs: dict, meal: dict, pre_meal_glucose: float | None = None, *,
            offset: float = 0.0, target_probability: float = DEFAULT_TARGET,
            max_edits: int = 4) -> dict:
    """Edits that lower this meal's predicted response, gentlest first.

    Returns an empty ``edits`` list when the meal is already at or below the
    target. An app that always has a suggestion is inventing one.
    """
    base = risk.score(labs, meal, pre_meal_glucose, offset=offset)

    if base["probability"] <= target_probability:
        return {
            "original": base,
            "edits": [],
            "target_probability": target_probability,
            "note": "This meal is already in the lower range of what this model "
                    "predicts for you.",
        }

    edits = []
    for description, changes in EDIT_LADDER:
        variant_meal, applied = apply_edit(meal, changes)
        if not applied:
            continue
        scored = risk.score(labs, variant_meal, pre_meal_glucose, offset=offset)
        if scored["probability"] >= base["probability"]:
            continue  # never surface a change that does not help
        edits.append({
            "description": description,
            "changes": applied,
            "resulting_meal": variant_meal,
            "probability": scored["probability"],
            "delta_probability": round(scored["probability"] - base["probability"], 3),
            "predicted_peak_mgdl": scored["predicted_peak_mgdl"],
            "predicted_iauc": scored["predicted_iauc"],
            "reaches_target": scored["probability"] <= target_probability,
        })
        if scored["probability"] <= target_probability:
            break

    note = ("Changes this model predicts would lower your response. "
            "Not dietary advice.")
    if edits and not any(e["reaches_target"] for e in edits):
        note = ("None of these reach the target on their own — they are the "
                "changes that help most. Not dietary advice.")
    elif not edits:
        note = ("No single change tried lowered the prediction for this meal.")

    return {
        "original": base,
        "edits": edits[:max_edits],
        "target_probability": target_probability,
        "note": note,
    }


def from_history(history: list[dict], meal_type: str, *,
                 threshold_mgdl: float | None = None, limit: int = 3) -> list[dict]:
    """The user's own past meals of this type that went well.

    This is what "previously successful meals" means in practice: their data,
    their outcomes, held on their device. Nothing from the study cohort is
    surfaced here.

    Only entries with a real observed outcome qualify — a past *prediction* is
    not evidence that a meal went well.
    """
    cutoff = threshold_mgdl if threshold_mgdl is not None else risk.spec()["threshold_mgdl"]
    scored = []
    for entry in history or []:
        meal = entry.get("meal") or {}
        if meal.get("meal_type") != meal_type:
            continue
        peak = entry.get("observed_peak")
        iauc = entry.get("observed_iauc")
        if peak is None and iauc is None:
            continue
        if peak is not None and peak >= cutoff:
            continue
        scored.append({
            "meal": meal,
            "observed_peak": peak,
            "observed_iauc": iauc,
            "logged_at": entry.get("logged_at"),
            "_sort": peak if peak is not None else float(iauc) + 1000,
        })
    scored.sort(key=lambda e: e["_sort"])
    for entry in scored:
        entry.pop("_sort", None)
    return scored[:limit]


def personal_offset(labs: dict, history: list[dict],
                    pre_meal_glucose: float | None = None) -> tuple[float, int]:
    """Shrunk correction from a user's logged outcomes. Returns (offset, k).

    Each historical meal is re-scored with the population model and compared to
    what actually happened; the mean gap, shrunk by k/(k+5), is the correction.
    """
    import personalize

    residuals = []
    for entry in history or []:
        meal = entry.get("meal") or {}
        observed = entry.get("observed_iauc")
        if observed is None or not meal.get("carbs") or not meal.get("meal_type"):
            continue
        try:
            predicted = risk.score(labs, meal, entry.get("pre_meal_glucose"))["predicted_iauc"]
        except (ValueError, KeyError):
            continue
        residuals.append(float(observed) - predicted)

    return personalize.offset_from_residuals(residuals), len(residuals)
