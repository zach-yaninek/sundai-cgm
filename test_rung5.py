"""Pins rung 5, the recommender and the personalisation loop.

Every assertion here is tied to something that would be silently wrong rather
than loudly broken: a calibrator that drifts, a serving vector built in the wrong
order, a "personalised" prediction with no evidence behind it, or a suggested
edit that does not actually help.

    python test_rung5.py
"""
import sys

import numpy as np

import personalize
import recommend
import risk

FAILURES = []


def ok(cond, msg):
    if not cond:
        FAILURES.append(msg)
    print(("PASS  " if cond else "FAIL  ") + msg)


HEALTHY = {"age": 30, "bmi": 22.0, "body_weight": 140, "height": 67,
           "a1c_pdl_lab": 5.2, "fasting_glu___pdl_lab": 88, "insulin": 4.0,
           "triglycerides": 70, "cholesterol": 175, "hdl": 65, "non_hdl": 110,
           "ldl_cal": 95, "vldl_cal": 14, "cho_hdl_ratio": 2.7}
RESISTANT = {"age": 55, "bmi": 31.0, "body_weight": 200, "height": 68,
             "a1c_pdl_lab": 6.2, "fasting_glu___pdl_lab": 115, "insulin": 18.0,
             "triglycerides": 190, "cholesterol": 210, "hdl": 38, "non_hdl": 172,
             "ldl_cal": 130, "vldl_cal": 38, "cho_hdl_ratio": 5.5}
MEAL = {"carbs": 66, "protein": 20, "fat": 18, "fiber": 4,
        "calories": 712, "meal_type": "dinner"}

# ------------------------------------------------------------------ artifacts
spec = risk.spec()
print("-- artifacts")
ok(spec["threshold_mgdl"] == 140.0, "threshold is 140 mg/dL")
ok(set(spec["variants"]) == {"with_glucose", "no_glucose"}, "both variants present")
ok("amount_consumed" not in spec["variants"]["with_glucose"]["columns"],
   "amount_consumed is excluded (mixed scales, see cgm.py NOTES 10)")
for variant in ("with_glucose", "no_glucose"):
    sv = spec["variants"][variant]
    for head in ("iauc", "peak_abs", "exceeds"):
        booster = risk._booster(variant, head)
        ok(booster.num_features() == len(sv["columns"]),
           f"{variant}/{head}: booster matches the spec's {len(sv['columns'])} columns")

# ---------------------------------------------------------------- calibration
print("\n-- calibration")
for variant in ("with_glucose", "no_glucose"):
    head = spec["variants"][variant]["heads"]["exceeds"]
    knots = head["calibrator"]
    grid = np.linspace(0, 1, 50)
    out = risk.apply_calibrator(knots, grid)
    ok(bool(np.all(np.diff(out) >= -1e-9)),
       f"{variant}: calibrator is monotone ({knots['method']})")
    ok(bool(np.all((out >= 0) & (out <= 1))), f"{variant}: calibrated output stays in [0,1]")
    ok(head["cv_ece"] <= min(v["ece"] for v in head["calibrator_comparison"].values()) + 1e-9,
       f"{variant}: the chosen calibrator has the lowest measured ECE")
    ok(head["cv_auc"] > 0.75, f"{variant}: held-out AUC {head['cv_auc']} clears 0.75")

# cohort percentile must be monotone in probability, or the band lies
pcts = [risk.cohort_percentile(p, "with_glucose") for p in np.linspace(0, 1, 25)]
ok(pcts == sorted(pcts), "cohort percentile is monotone in probability")

# ------------------------------------------------------------------- scoring
print("\n-- scoring")
hi = risk.score(RESISTANT, MEAL, 104)
lo = risk.score(HEALTHY, MEAL, 104)
ok(hi["probability"] > lo["probability"],
   f"insulin-resistant scores higher ({hi['probability']} vs {lo['probability']})")
ok(risk.score(RESISTANT, {**MEAL, "carbs": 15, "calories": 400}, 104)["probability"]
   < hi["probability"], "less carbohydrate lowers the probability")
ok(risk.score(RESISTANT, MEAL, 104)["variant"] == "with_glucose"
   and risk.score(RESISTANT, MEAL)["variant"] == "no_glucose",
   "variant is chosen by whether a glucose reading was supplied")
ok(risk.score(RESISTANT, MEAL, 104)["confidence_band"] == "narrow"
   and risk.score({"a1c_pdl_lab": 6.2}, MEAL)["confidence_band"] == "wide",
   "confidence band widens as inputs are withheld")
ok(risk.score(HEALTHY, {**MEAL, "carbs": 0, "calories": 5}, 90)["predicted_iauc"] >= 0,
   "iAUC is never negative")

# the derived feature must not depend on the caller knowing the formula
a = risk.build_row(RESISTANT, MEAL, 104)[0]
b = risk.build_row({**RESISTANT, "homa_ir": 115 * 18.0 / 405}, MEAL, 104)[0]
ok(np.allclose(a, b), "homa_ir is derived from glucose x insulin / 405")

for bad, why in [
    (lambda: risk.score(RESISTANT, {**MEAL, "meal_type": "brunch"}, 104), "unknown meal_type raises"),
    (lambda: risk.score({"ldl": 100}, MEAL, 104), "unknown lab field raises"),
    (lambda: risk.score(RESISTANT, {"carbs": 10, "meal_type": "dinner"}, 104), "missing calories raises"),
]:
    try:
        bad()
        ok(False, why)
    except ValueError:
        ok(True, why)

# ---------------------------------------------------------- personalisation
print("\n-- personalisation")
ok(personalize.shrinkage(0) == 0.0, "no history applies no correction")
ok(abs(personalize.shrinkage(5) - 0.5) < 1e-9, "k=5 gives half weight")
ok(all(personalize.shrinkage(k) < personalize.shrinkage(k + 1) for k in range(0, 20)),
   "shrinkage rises monotonically with logged meals")
ok(personalize.offset_from_residuals([]) == 0.0, "empty history yields a zero offset")
ok(abs(personalize.offset_from_residuals([10.0] * 5) - 5.0) < 1e-9,
   "five residuals of +10 give an offset of +5 (half weight)")
ok(abs(personalize.offset_from_residuals([10.0]) - 10.0 / 6) < 1e-9,
   "one residual moves the offset only 1/6 of the way")

base = risk.score(RESISTANT, MEAL, 104)
warm = risk.score(RESISTANT, MEAL, 104, offset=20.0)
ok(warm["predicted_iauc"] > base["predicted_iauc"], "a positive offset raises the estimate")
ok(risk.score(RESISTANT, MEAL, 104, offset=0.0) == base,
   "a zero offset reproduces the population prediction exactly")

hist = [{"meal": MEAL, "observed_iauc": 120.0, "pre_meal_glucose": 104}] * 4
offset, k = recommend.personal_offset(RESISTANT, hist, 104)
ok(k == 4 and offset > 0, f"a consistently under-predicted history gives a positive offset ({offset:+.1f})")
ok(recommend.personal_offset(RESISTANT, [], 104) == (0.0, 0),
   "no usable history gives no offset")
ok(recommend.personal_offset(RESISTANT, [{"meal": MEAL}], 104)[1] == 0,
   "an entry with no observed outcome is not evidence")

# --------------------------------------------------------------- recommender
print("\n-- recommender")
result = recommend.suggest(RESISTANT, MEAL, 104)
ok(len(result["edits"]) > 0, f"a high-risk meal gets suggestions ({len(result['edits'])})")
ok(all(e["delta_probability"] < 0 for e in result["edits"]),
   "every returned edit lowers the probability")
ok(all(e["resulting_meal"]["calories"] != MEAL["calories"]
       for e in result["edits"] if "carbs" in e["changes"]),
   "a carbohydrate edit moves calories too — no impossible meals")
gentle = recommend.suggest(HEALTHY, {**MEAL, "carbs": 12, "calories": 260}, 104)
ok(gentle["edits"] == [], "an already-low-risk meal returns no edits")
ok("already in the lower range" in gentle["note"], "and says so rather than inventing one")

past = recommend.from_history(
    [{"meal": MEAL, "observed_peak": 120},
     {"meal": MEAL, "observed_peak": 190},
     {"meal": {**MEAL, "meal_type": "lunch"}, "observed_peak": 110},
     {"meal": MEAL}],
    "dinner")
ok(len(past) == 1 and past[0]["observed_peak"] == 120,
   "history surfaces only same-type meals with an observed outcome below threshold")

print("\n" + "=" * 60)
if FAILURES:
    print(f"{len(FAILURES)} FAILED:")
    for msg in FAILURES:
        print("  -", msg)
    sys.exit(1)
print("all checks passed")
sys.exit(0)
