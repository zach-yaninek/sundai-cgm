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

# The history form asks what glucose PEAKED at, because that is the number a
# person can read off a CGM; the offset is defined on iAUC. An entry carrying
# only a peak used to be skipped, so k stayed 0 and the app never personalised
# however much was logged, while the UI counted the meals up. These pin the
# conversion, because nothing failed when it was missing.
peak_hist = [{"meal": MEAL, "observed_peak": 195.0, "pre_meal_glucose": 104}] * 3
peak_offset, peak_k = recommend.personal_offset(RESISTANT, peak_hist, 104)
ok(peak_k == 3, f"an outcome logged as a peak is evidence, not a skipped row (k={peak_k})")
ok(peak_offset > 0,
   f"peaks above what was predicted raise the estimate ({peak_offset:+.1f})")

low_hist = [{"meal": MEAL, "observed_peak": 118.0, "pre_meal_glucose": 104}] * 3
low_offset, low_k = recommend.personal_offset(RESISTANT, low_hist, 104)
ok(low_k == 3 and low_offset < 0,
   f"peaks below what was predicted lower it ({low_offset:+.1f})")

# A measured iAUC must still win outright: converting is the fallback, and a
# reading that needs no conversion should not be routed through one.
_scored = risk.score(RESISTANT, MEAL, 104)
ok(recommend.observed_iauc({"observed_iauc": 42.0, "observed_peak": 300.0},
                           RESISTANT, _scored) == 42.0,
   "a measured iAUC is used as-is, never re-derived from the peak")
ok(recommend.observed_iauc({"observed_peak": 90.0, "pre_meal_glucose": 104},
                           RESISTANT, _scored) == 0.0,
   "a peak below baseline is a real observation of no rise, clipped at zero")
ok(recommend.observed_iauc({}, RESISTANT, _scored) is None,
   "an entry with neither reading converts to nothing")

# --- two-parameter calibration -------------------------------------------
# The gate is the load-bearing part. personalize_compare.py measured a
# two-parameter fit as clearly WORSE than a flat offset below k=6 (28.00 against
# 25.77 at k=3) and better from k=6 on, so a regression that moves the gate is a
# regression in accuracy that nothing else would catch.
import shrinkage as shrink

flat = shrink.fit_calibration([100.0] * 4, [120.0] * 4)
ok(not flat.learned_slope and flat.k == 4,
   "below the gate only an intercept is learned")
ok(abs(flat.offset_for(50.0) - flat.offset_for(500.0)) < 1e-9,
   "an intercept-only correction is the same whatever it corrects")
ok(abs(flat.offset_for(100.0) - shrink.offset_from_residuals([20.0] * 4)) < 1e-9,
   "the intercept-only regime reproduces the old scalar offset exactly")

rising = shrink.fit_calibration([50.0, 80.0, 110.0, 140.0, 170.0, 200.0],
                                [55.0, 95.0, 135.0, 180.0, 220.0, 265.0])
ok(rising.learned_slope and rising.k == 6,
   f"at the gate a slope is learned (slope={rising.slope:.2f})")
ok(rising.offset_for(200.0) > rising.offset_for(60.0),
   "a slope corrects large predictions more than small ones")

ok(shrink.fit_calibration([], []).offset_for(120.0) == 0.0,
   "no history corrects nothing")
ok(not shrink.fit_calibration([100.0] * 8, [130.0] * 8).learned_slope,
   "predictions with no spread fall back to intercept-only, not a wild slope")

# End to end through the serving path, since that is where it has to hold.
cal_hist = [{"meal": {**MEAL, "carbs": c}, "observed_iauc": o, "pre_meal_glucose": 104}
            for c, o in [(20, 30.0), (40, 70.0), (60, 120.0),
                         (80, 180.0), (100, 240.0), (120, 300.0)]]
cal = recommend.personal_calibration(RESISTANT, cal_hist, 104)
ok(cal.k == 6 and cal.learned_slope,
   f"six logged meals through the API path learn a slope (k={cal.k})")
small = risk.score(RESISTANT, {**MEAL, "carbs": 20, "calories": 400}, 104, calibration=cal)
large = risk.score(RESISTANT, {**MEAL, "carbs": 120, "calories": 1100}, 104, calibration=cal)
ok(abs(large["offset_applied"]) > abs(small["offset_applied"]),
   f"the correction scales with the prediction "
   f"({small['offset_applied']:+.1f} vs {large['offset_applied']:+.1f})")
ok(risk.score(RESISTANT, MEAL, 104, calibration=shrink.Calibration()) == base,
   "an empty calibration reproduces the population prediction exactly")

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

# --------------------------------------------------- value of information
print("\n-- value of information")
import json as _json
from pathlib import Path as _Path

import sundai_cgm

_repo = _json.loads((_Path("artifacts") / "information_value.json").read_text())
_pkg = _json.loads((_Path("sundai_cgm") / "_data" / "information_value.json").read_text())
ok(_repo == _pkg, "the shipped grid is identical to the generated one (no drift)")

blind = sundai_cgm.value_of_information({}, 104)
covered = sundai_cgm.value_of_information(
    {"a1c_pdl_lab": 6.2, "insulin": 18.0, "fasting_glu___pdl_lab": 115}, 104)
ok(blind["score"] == 1.0, "no labs at all scores 1.0 — a draw recovers the whole gap")
ok(covered["score"] == 0.0,
   "a recent core panel scores 0.0 — a redraw would add nothing this model can use")
ok(covered["recommended_panel"] is None,
   "and no panel is recommended, rather than inventing one")
ok(blind["auc_after_draw"] > blind["auc_now"], "the quoted gain is a real AUC gap")

ok(sundai_cgm.best_tier(_repo["grid"]["with_glucose"]) == "core",
   "core is preferred over full — the lipid panel adds ~0.002 AUC")
ok(sundai_cgm.best_tier(_repo["grid"]["no_glucose"]) == "core",
   "and without a CGM the full panel is actually worse than core")
ok(sundai_cgm.tier_for({"hdl": 55}) == "none",
   "a lone lipid value is not a metabolic panel")
ok(0.8 < sundai_cgm.reliability() <= 1.0,
   f"reliability weight is the measured best AUC ({sundai_cgm.reliability():.3f})")

# The score must fall as more is known — a monotonicity the fusion layer relies on.
_partial = sundai_cgm.value_of_information({"a1c_pdl_lab": 6.2}, 104)
ok(blind["score"] >= _partial["score"] >= covered["score"],
   "score falls monotonically as more analytes become known")

# --------------------------------------------------------------- explanations
print("\n-- explanations (guardrails, no credentials needed)")
import explain

_attr = explain.drivers(RESISTANT, MEAL, 104)
_assess = risk.score(RESISTANT, MEAL, 104)
_payload = explain.build_payload(_assess, _attr, meal_type="dinner")

ok(len(_attr) > 0 and all("contribution" in d for d in _attr),
   f"real SHAP attributions are produced ({len(_attr)} drivers)")
ok(all("value" not in d for d in _attr),
   "attributions carry no feature values, only magnitudes")

# The privacy property: no lab or macro value may appear as a token upstream.
import json as _js
import re as _re
_tokens = set(_re.findall(r"\d+(?:\.\d+)?", _js.dumps(_payload)))
_secret = {**RESISTANT, **{k: v for k, v in MEAL.items() if k != "meal_type"}}
_leaked = [k for k, v in _secret.items()
           if isinstance(v, (int, float)) and v > 10
           and (str(v) in _tokens or str(float(v)) in _tokens or str(int(v)) in _tokens)]
ok(not _leaked, f"payload leaks no lab or macro value upstream (found {_leaked})")

_clean = {"headline": "This model puts the chance above 140 mg/dL at 90%.",
          "drivers": ["carbohydrate raises it"], "caveat": "Fitted to 45 people."}
ok(explain.validate(_clean, _payload)[0], "a clean response validates")
for _bad, _why in [
    ({**_clean, "headline": "Your HbA1c of 6.2 drives this."}, "an invented number is rejected"),
    ({**_clean, "caveat": "This is a dangerous meal."}, "'dangerous' is rejected"),
    ({**_clean, "drivers": ["you should eat less rice"]}, "dietary advice is rejected"),
    ({"headline": "x", "drivers": []}, "a malformed response is rejected"),
]:
    ok(not explain.validate(_bad, _payload)[0], _why)

_template = explain.template_explanation(_payload)
ok(explain.validate(_template, _payload)[0],
   "the fallback template passes its own validation")
ok("140" in _template["headline"] and "45" in _template["caveat"],
   "the fallback names the threshold and the cohort size")

# With no client and no credentials, explain() must still return something usable.
_result = explain.explain(RESISTANT, MEAL, 104, _assess)
ok(_result["source"] == "template" if not explain.available() else True,
   f"explain() degrades to the template without credentials (source={_result['source']})")
ok(bool(_result["headline"]) and bool(_result["caveat"]),
   "explain() always returns a usable explanation")


class _FakeClient:
    """Stands in for the SDK so the live path is testable without a key."""

    def __init__(self, payload):
        self._payload = payload
        self.messages = self

    def create(self, **kwargs):
        import types
        block = types.SimpleNamespace(type="text", text=_js.dumps(self._payload))
        return types.SimpleNamespace(content=[block], stop_reason="end_turn")


explain._CACHE.clear()
_good = explain.explain(RESISTANT, MEAL, 104, _assess, client=_FakeClient(_clean))
ok(_good["source"] == "claude", "a valid live response is used")

explain._CACHE.clear()
_bad_live = explain.explain(RESISTANT, MEAL, 104, _assess,
                            client=_FakeClient({**_clean, "caveat": "dangerous meal"}))
ok(_bad_live["source"] == "template" and "rejected_reason" in _bad_live,
   f"a live response that breaks a rule is discarded ({_bad_live.get('rejected_reason')})")

print("\n" + "=" * 60)
if FAILURES:
    print(f"{len(FAILURES)} FAILED:")
    for msg in FAILURES:
        print("  -", msg)
    sys.exit(1)
print("all checks passed")
sys.exit(0)
