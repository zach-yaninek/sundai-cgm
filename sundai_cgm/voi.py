"""Value of drawing labs, as a lookup over a measured grid.

Zero heavy dependencies on purpose — this is stdlib only, so a consumer can
answer "is a blood draw worth it for this person" without pulling in xgboost or
the model artifacts.

The grid in `_data/information_value.json` is produced by the repository's
`information_value.py` under cold-start, subject-grouped cross-validation. It
records how well the meal-risk flag performs given what is known about someone:
whether they have a continuous glucose reading, and which tier of blood panel
they have had drawn.

What the numbers say, and why the framing matters
-------------------------------------------------
A CGM measures interstitial glucose and nothing else. It cannot produce HbA1c,
fasting insulin or a lipid panel, and HOMA-IR needs a needle. So a draw still
adds information to someone already wearing one — measured, that is flag AUC
0.846 -> 0.887.

But almost all of it comes from three analytes. The full lipid panel adds ~0.002
AUC on top of HbA1c, insulin and fasting glucose, and without a CGM reading the
full panel is actually *worse* than the core three (0.835 vs 0.842) — more
features against the same 45 subjects. So this module recommends the cheapest
tier that performs within tolerance of the best, which is the core draw, rather
than reflexively asking for a full panel.

This is a statement about information gain, not about detecting disease.
"""
from __future__ import annotations

import json
from importlib import resources

# Obtainable without a blood draw.
FREE_FIELDS = ("age", "bmi", "body_weight", "height")

TIERS = {
    "none": [],
    "core": ["a1c_pdl_lab", "insulin", "fasting_glu___pdl_lab", "homa_ir"],
    "full": ["a1c_pdl_lab", "insulin", "fasting_glu___pdl_lab", "homa_ir",
             "triglycerides", "cholesterol", "hdl", "non_hdl",
             "ldl_cal", "vldl_cal", "cho_hdl_ratio"],
}

# A tier only wins if it beats the cheaper one by more than this. 0.005 AUC on
# 1,382 meals from 45 subjects is well inside the noise.
TOLERANCE = 0.005

_GRID: dict | None = None


def grid() -> dict:
    """The measured performance grid, keyed by glucose-reading then lab tier."""
    global _GRID
    if _GRID is None:
        raw = resources.files("sundai_cgm").joinpath("_data/information_value.json")
        _GRID = json.loads(raw.read_text())
    return _GRID


def tier_for(labs: dict | None) -> str:
    """Which measured tier this person's known lab fields correspond to."""
    known = {k for k, v in (labs or {}).items() if v is not None}
    if not known & set(TIERS["core"]):
        return "none"
    lipids = set(TIERS["full"]) - set(TIERS["core"])
    # One stray HDL does not make a full panel.
    return "full" if len(known & lipids) >= len(lipids) - 1 else "core"


def best_tier(cell: dict, *, tolerance: float = TOLERANCE) -> str:
    """Cheapest tier performing within ``tolerance`` of the best measured AUC."""
    ranked = ["none", "core", "full"]
    best = max(cell[t]["auc"] for t in ranked)
    for tier in ranked:
        if cell[tier]["auc"] >= best - tolerance:
            return tier
    return "full"


def value_of_information(labs: dict | None = None,
                         pre_meal_glucose: float | None = None) -> dict:
    """How much would drawing labs sharpen this person's predictions?

    Returns a 0-1 ``score`` suitable as fusion evidence, plus the reasoning:
    which tier they are on, which is worth reaching, what the AUC gap is, and
    exactly which analytes are missing.

    ``score`` of 1.0 means the model is running blind on labs and a draw recovers
    the whole measured gap. 0.0 means they already have everything this model can
    use, and a redraw would add nothing — which is the honest answer for someone
    with a recent panel, however much a dashboard would prefer a number to show.
    """
    payload = grid()
    key = "with_glucose" if pre_meal_glucose is not None else "no_glucose"
    cell = payload["grid"][key]

    current_tier = tier_for(labs)
    target_tier = best_tier(cell)

    current = cell[current_tier]["auc"]
    target = cell[target_tier]["auc"]
    floor = cell["none"]["auc"]
    span = target - floor

    gain = max(0.0, target - current)
    score = min(max(gain / span, 0.0), 1.0) if span > 1e-9 else 0.0

    missing = [f for f in TIERS[target_tier]
               if f != "homa_ir" and (labs or {}).get(f) is None]

    if not missing:
        panel, reason = None, (
            "A recent panel is already on file; drawing again would not improve "
            "what this model can say about their meals."
        )
    else:
        panel = ("HbA1c, fasting insulin and fasting glucose"
                 if target_tier == "core" else "full metabolic and lipid panel")
        reason = (
            f"{len(missing)} of the analytes this model relies on "
            f"{'is' if len(missing) == 1 else 'are'} unknown. "
            f"Drawing {panel} would move the meal-response flag from "
            f"AUC {current:.3f} to {target:.3f}"
            + (" for someone already wearing a CGM."
               if pre_meal_glucose is not None else
               ", and this person has no glucose reading either.")
        )

    return {
        "score": round(score, 3),
        "current_tier": current_tier,
        "recommended_tier": target_tier,
        "auc_now": current,
        "auc_after_draw": target,
        "auc_gain": round(gain, 4),
        "missing_fields": missing,
        "recommended_panel": panel,
        "used_pre_meal_glucose": pre_meal_glucose is not None,
        "reason": reason,
    }


def reliability() -> float:
    """How much a consumer should weight this expert.

    The best measured AUC for the flag, so the weight reflects something that was
    actually validated rather than a number chosen to feel about right.
    """
    payload = grid()
    return max(cell[t]["auc"]
               for cell in payload["grid"].values()
               for t in ("none", "core", "full"))
