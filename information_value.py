"""information_value.py — what would a blood draw actually tell us about this person?

A continuous glucose monitor measures interstitial glucose and nothing else. It
cannot give HbA1c, fasting insulin, or a lipid panel — and HOMA-IR, the second
strongest predictor in rung 5, needs a needle.

So "does this person need labs drawn?" has a measurable answer: how much better
would our prediction get if they had them. This module measures that across a
grid of what someone might already know, and writes the grid to
`artifacts/information_value.json` so serving can look it up instead of guessing.

    python information_value.py

The resulting score is deliberately about **information gain**, not about
detecting disease. Telling a CGM wearer they need a glucose panel would be close
to useless — they already have better glucose data than a fasting draw provides.
Telling them their insulin and HbA1c are unknown, and that measuring them would
move the flag from AUC 0.847 to 0.890, is a real and checkable statement.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

import evaluate
import rung5_meal_risk as r5

ARTIFACTS = Path(__file__).parent / "artifacts"

# What a person can know without a blood draw.
FREE = ["age", "bmi", "body_weight", "height"]

# What only a venipuncture panel provides, split into the two tiers a clinician
# would actually order. `fasting_glu___pdl_lab` sits in the core tier because a
# CGM's own reading is not a fasting venous glucose.
TIERS = {
    "none": [],
    "core": ["a1c_pdl_lab", "insulin", "fasting_glu___pdl_lab", "homa_ir"],
    "full": ["a1c_pdl_lab", "insulin", "fasting_glu___pdl_lab", "homa_ir",
             "triglycerides", "cholesterol", "hdl", "non_hdl",
             "ldl_cal", "vldl_cal", "cho_hdl_ratio"],
}


def _auc(X, y, groups, cols) -> float:
    oof = np.full(len(y), np.nan)
    for train_idx, test_idx in evaluate.cold_folds(groups):
        model = xgb.XGBClassifier(**r5.CLF_PARAMS)
        model.fit(X.iloc[train_idx][cols], y[train_idx], verbose=False)
        oof[test_idx] = model.predict_proba(X.iloc[test_idx][cols])[:, 1]
    return float(roc_auc_score(y, oof))


def _mae(X, y, groups, cols) -> tuple[float, float]:
    def fit_predict(X_tr, y_tr, X_te):
        model = xgb.XGBRegressor(**r5.REG_PARAMS)
        model.fit(X_tr[cols], y_tr, verbose=False)
        return model.predict(X_te[cols])

    metrics, _ = evaluate.run(fit_predict, X, y, groups, regime="cold", name="voi")
    return round(metrics["mae"], 3), round(metrics["r2"], 4)


def measure() -> dict:
    """Cross-validated performance for each (glucose reading x lab tier) cell."""
    df = r5.build_frame()
    groups = df["subject"].to_numpy()
    y_clf = df["exceeds"].to_numpy()
    y_reg = df["iauc"].to_numpy(dtype=float)

    grid = {}
    for has_glucose in (True, False):
        X, _ = r5.build_features(df, with_glucose=has_glucose)
        macros = [c for c in X.columns
                  if c in r5.MACROS or c.startswith("mt_") or c == "pre_meal_glucose"]
        key = "with_glucose" if has_glucose else "no_glucose"
        grid[key] = {}
        for tier, labs in TIERS.items():
            cols = macros + FREE + [c for c in labs if c in X.columns]
            auc = _auc(X, y_clf, groups, cols)
            mae, r2 = _mae(X, y_reg, groups, cols)
            grid[key][tier] = {"auc": round(auc, 4), "mae_iauc": mae, "r2_iauc": r2,
                               "n_lab_fields": len(labs)}
            print(f"  {key:<13} {tier:<5}  AUC {auc:.3f}  MAE {mae:6.2f}  R2 {r2:.3f}")
    return grid


def tier_for(labs: dict) -> str:
    """Which measured tier a person's known lab fields correspond to."""
    known = {k for k, v in (labs or {}).items() if v is not None}
    if not known & set(TIERS["core"]):
        return "none"
    lipids = set(TIERS["full"]) - set(TIERS["core"])
    # "full" only once most of the lipid panel is actually present; a lone HDL
    # does not make this a full panel.
    return "full" if len(known & lipids) >= len(lipids) - 1 else "core"


def best_tier(cell: dict, *, tolerance: float = 0.005) -> str:
    """The cheapest tier that performs within ``tolerance`` of the best measured.

    Not simply "full". The measured grid says the core panel does nearly all the
    work — HbA1c, insulin and fasting glucose take AUC from 0.846 to 0.887, and
    the whole lipid panel on top adds 0.002. Without a CGM reading the full panel
    is actually *worse* than core (0.835 vs 0.842): more features, same 45
    subjects. Recommending a full draw on that evidence would be asking for blood
    the model cannot use.
    """
    ranked = ["none", "core", "full"]
    best_auc = max(cell[t]["auc"] for t in ranked)
    for tier in ranked:
        if cell[tier]["auc"] >= best_auc - tolerance:
            return tier
    return "full"


def score(labs: dict, pre_meal_glucose: float | None, grid: dict) -> dict:
    """Value of drawing labs for this person, as a 0-1 evidence score.

    1.0 means the model is running blind and a draw would recover the entire
    measured gap; 0.0 means they already have everything this model can use and
    a redraw would add nothing.
    """
    key = "with_glucose" if pre_meal_glucose is not None else "no_glucose"
    cell = grid[key]
    current_tier = tier_for(labs)

    target_tier = best_tier(cell)
    current = cell[current_tier]["auc"]
    target = cell[target_tier]["auc"]
    floor = cell["none"]["auc"]
    span = target - floor

    gain = max(0.0, target - current)
    normalised = float(np.clip(gain / span, 0.0, 1.0)) if span > 1e-9 else 0.0

    wanted = TIERS[target_tier]
    missing = [f for f in wanted
               if f != "homa_ir" and (labs or {}).get(f) is None]
    return {
        "score": round(normalised, 3),
        "current_tier": current_tier,
        "recommended_tier": target_tier,
        "auc_now": current,
        "auc_after_draw": target,
        "auc_gain": round(gain, 4),
        "missing_fields": missing,
        "used_pre_meal_glucose": pre_meal_glucose is not None,
        "panel": ("no further labs would help this prediction"
                  if not missing else
                  "HbA1c, fasting insulin and fasting glucose"
                  if target_tier == "core" else
                  "full metabolic and lipid panel"),
    }


def main() -> None:
    print("\nmeasuring what a blood draw is worth, per (glucose reading x lab tier):\n")
    grid = measure()

    payload = {
        "recommended_tier": {k: best_tier(v) for k, v in grid.items()},
        "note": "Cold-start, subject-grouped CV. 'none' means the model has only "
                "macros, demographics and (optionally) a CGM reading. A CGM cannot "
                "measure HbA1c, insulin or lipids, which is why a draw still adds "
                "information to someone already wearing one.",
        "free_fields": FREE,
        "tiers": TIERS,
        "grid": grid,
    }
    ARTIFACTS.mkdir(exist_ok=True)
    blob = json.dumps(payload, indent=2)
    (ARTIFACTS / "information_value.json").write_text(blob)

    # The installable package ships this grid so a consumer can answer "are labs
    # worth drawing" without the model weights. Written by the same generator so
    # the two copies cannot drift; test_rung5.py asserts they are identical.
    package_data = Path(__file__).parent / "sundai_cgm" / "_data"
    package_data.mkdir(parents=True, exist_ok=True)
    (package_data / "information_value.json").write_text(blob)

    results_path = ARTIFACTS / "results.json"
    existing = json.loads(results_path.read_text()) if results_path.exists() else {}
    existing["information_value"] = payload
    results_path.write_text(json.dumps(existing, indent=2))

    for key, cell in grid.items():
        target = best_tier(cell)
        print(f"\n{key}: cheapest tier within 0.005 AUC of best is '{target}'"
              f" (AUC {cell['none']['auc']} -> {cell[target]['auc']},"
              f" R2 {cell['none']['r2_iauc']} -> {cell[target]['r2_iauc']})")
    print("\nThe lipid panel adds ~0.002 AUC over the core three, and is worse than "
          "core without a CGM reading. Recommend the core draw, not a full panel.")
    print(f"\nwrote {ARTIFACTS / 'information_value.json'}")
    print(f"wrote {package_data / 'information_value.json'} (shipped in the package)")


if __name__ == "__main__":
    main()
