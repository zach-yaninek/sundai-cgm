"""Round-trip test for the serving path.

The point of this file is narrow and important: prove that `predict.py`
reconstructs the feature vector *identically* to how training built it. Skew
between the two is the classic way a deployed model produces confident nonsense
— nothing errors, the numbers just quietly stop meaning anything.

So the test reloads the saved booster through the public serving API and checks
it reproduces the in-sample predictions recorded at training time, row for row.

    python test_predict.py
"""
import sys

import numpy as np
import pandas as pd

import cgm
import predict
import rung4_subject as r4

FAILURES = []


def ok(cond, msg):
    if not cond:
        FAILURES.append(msg)
    print(("PASS  " if cond else "FAIL  ") + msg)


# ----------------------------------------------------------------- artifacts
s = predict.spec()
print("-- spec")
ok(s["model"] == "rung4_subject.json", "spec names the served model")
ok(len(s["columns"]) == len(s["meal_columns"]) + len(s["lab_columns"]),
   f"columns = meal + lab, no gut ({len(s['columns'])} total)")
ok(not any(c in s["columns"] for c in s["gut_columns_not_served"]),
   "gut columns are excluded from the served feature set")
ok(predict.model().num_features() == len(s["columns"]),
   "booster feature count matches the spec")
ok("self_identify" in s["excluded_features"],
   "race/ethnicity is recorded as deliberately excluded")

catalog = predict.dish_catalog()
print("\n-- catalog")
ok(len(catalog) == 16, f"16 standardized dishes (got {len(catalog)})")
ok(catalog["n_subjects"].max() >= 43, "top dish was eaten by 43+ subjects")

# ------------------------------------------------------------- ROUND TRIP
print("\n-- round trip: artifacts-only serving reproduces training predictions")
insample = pd.read_parquet(predict.ARTIFACTS / "rung4_insample.parquet")
bio = cgm.bio().set_index("subject")

rows, expected = [], []
sample = insample.sample(min(200, len(insample)), random_state=0)
for row in sample.itertuples():
    panel = {c: bio.loc[row.subject, c] for c in r4.LAB_FEATURES}
    vec, imputed = predict.build_row(panel, row.meal_id)
    if imputed:  # a real panel should need no imputation
        FAILURES.append(f"unexpected imputation for subject {row.subject}: {imputed}")
    rows.append(vec[0])
    expected.append(row.pred_rung4_insample)

got = predict._predict_raw(np.vstack(rows).astype(np.float32))
exp = np.clip(np.asarray(expected, dtype=float), 0.0, None)  # serving clips at 0
diff = np.abs(got - exp)
ok(diff.max() < 1e-4,
   f"{len(sample)} rows reproduce to <1e-4 (max diff {diff.max():.2e})")

# ------------------------------------------------------------------ behaviour
print("\n-- behaviour")
healthy = {"age": 30, "bmi": 22.0, "a1c_pdl_lab": 5.2,
           "fasting_glu___pdl_lab": 88, "insulin": 4.0}
resistant = {"age": 55, "bmi": 31.0, "a1c_pdl_lab": 6.2,
             "fasting_glu___pdl_lab": 115, "insulin": 18.0}

rh = predict.rank_meals(healthy)
rr = predict.rank_meals(resistant)
ok(rr["predicted_iauc"].mean() > rh["predicted_iauc"].mean(),
   f"insulin-resistant panel predicts higher overall "
   f"({rr.predicted_iauc.mean():.0f} vs {rh.predicted_iauc.mean():.0f})")
ok((rh["predicted_iauc"] >= 0).all() and (rr["predicted_iauc"] >= 0).all(),
   "no negative iAUC is ever returned")
ok(rh["predicted_iauc"].is_monotonic_increasing, "rank_meals sorts ascending by default")
ok(len(rh) == 16, "every dish is ranked")

# HOMA-IR is derived when its inputs are present
vec_auto, _ = predict.build_row(resistant, catalog["meal_id"].iloc[0])
vec_expl, _ = predict.build_row({**resistant, "homa_ir": 115 * 18.0 / 405},
                                catalog["meal_id"].iloc[0])
ok(np.allclose(vec_auto, vec_expl), "homa_ir is derived from glucose x insulin / 405")

# partial panels degrade rather than explode
partial = predict.predict_response({"fasting_glu___pdl_lab": 95}, catalog["meal_id"].iloc[0])
ok(len(partial["imputed_fields"]) > 0, "a partial panel reports which fields were imputed")
ok(partial["predicted_iauc"] >= 0, "a partial panel still returns a usable number")

detail = predict.predict_response(resistant, catalog["meal_id"].iloc[0])
ok(len(detail["curve"]) == 25, f"curve is on a 25-point grid (got {len(detail['curve'])})")
ok("cohort" in detail["curve_note"], "the curve is labelled as an observed shape")

# ------------------------------------------------------------------- errors
print("\n-- errors are loud, not silent")
for bad, why in [
    (lambda: predict.build_row(healthy, "not-a-dish"), "unknown meal_id raises"),
    (lambda: predict.build_row({"ldl": 100}, catalog["meal_id"].iloc[0]),
     "unknown lab field raises rather than being ignored"),
]:
    try:
        bad()
        ok(False, why)
    except ValueError:
        ok(True, why)

print("\n" + "=" * 60)
if FAILURES:
    print(f"{len(FAILURES)} FAILED:")
    for msg in FAILURES:
        print("  -", msg)
    sys.exit(1)
print("all checks passed")
sys.exit(0)
