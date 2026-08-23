"""Validates API responses against contract/openapi.json.

This is the test that keeps the frontend honest. The collaborator builds against
`stub_server.py`; the real backend lands later in `serve.py`. Both are checked
against the *same* schema here, so a screen built against fixtures cannot break
when the model replaces them.

    python test_contract.py            # stub only (default)
    python test_contract.py --real     # also check serve.py once it exists
"""
import sys

import jsonschema

sys.path.insert(0, "contract")

import json
from pathlib import Path

from fastapi.testclient import TestClient

SPEC = json.loads((Path(__file__).parent / "contract" / "openapi.json").read_text())
FAILURES = []


def ok(cond, msg):
    if not cond:
        FAILURES.append(msg)
    print(("PASS  " if cond else "FAIL  ") + msg)


def validate(payload, schema_name, label):
    """Check one response body against a named schema in the spec."""
    schema = {**SPEC["components"]["schemas"][schema_name],
              "components": SPEC["components"],
              "$defs": SPEC["components"]["schemas"]}
    resolver = jsonschema.RefResolver.from_schema(
        {"components": SPEC["components"]}, store={"": {"components": SPEC["components"]}}
    )
    try:
        jsonschema.validate(payload, schema, resolver=resolver)
        ok(True, f"{label} validates against {schema_name}")
    except jsonschema.ValidationError as exc:
        ok(False, f"{label} fails {schema_name}: {exc.message} at {list(exc.absolute_path)}")


LABS_FULL = {
    "age": 55, "bmi": 31.0, "body_weight": 200, "height": 68,
    "a1c_pdl_lab": 6.2, "fasting_glu___pdl_lab": 115, "insulin": 18.0,
    "triglycerides": 190, "cholesterol": 210, "hdl": 38, "non_hdl": 172,
    "ldl_cal": 130, "vldl_cal": 38, "cho_hdl_ratio": 5.5,
}
LABS_SPARSE = {"a1c_pdl_lab": 5.2, "fasting_glu___pdl_lab": 88}
MEAL = {"carbs": 66, "protein": 20, "fat": 18, "fiber": 4,
        "calories": 712, "meal_type": "dinner"}


def check(client, name):
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")

    validate(client.get("/api/meta").json(), "Meta", f"{name} /api/meta")
    validate(client.get("/api/fields").json(), "FieldList", f"{name} /api/fields")

    body = {"labs": LABS_FULL, "meal": MEAL, "pre_meal_glucose": 104, "history": []}
    assess = client.post("/api/assess", json=body).json()
    validate(assess, "AssessResponse", f"{name} /api/assess")

    alts = client.post("/api/alternatives", json=body).json()
    validate(alts, "AlternativesResponse", f"{name} /api/alternatives")

    # Every confidence band must be reachable, or the UI cannot be built for it.
    bands = set()
    for labs, pre in [(LABS_FULL, 104), (LABS_FULL, None), (LABS_SPARSE, None), ({}, None)]:
        r = client.post("/api/assess", json={"labs": labs, "meal": MEAL,
                                             "pre_meal_glucose": pre}).json()
        bands.add(r["confidence"]["band"])
    ok(bands == {"narrow", "moderate", "wide"},
       f"{name}: all three confidence bands reachable (got {sorted(bands)})")

    # Direction sanity — fixtures must move the way the real model moves, or the
    # frontend gets built around behaviour that later reverses.
    worse = client.post("/api/assess", json={"labs": LABS_FULL, "meal": MEAL,
                                             "pre_meal_glucose": 104}).json()
    better = client.post("/api/assess", json={
        "labs": {**LABS_FULL, "a1c_pdl_lab": 5.0, "fasting_glu___pdl_lab": 85, "insulin": 3.0},
        "meal": MEAL, "pre_meal_glucose": 92}).json()
    ok(better["exceeds_140"]["probability"] < worse["exceeds_140"]["probability"],
       f"{name}: a healthier panel lowers the probability")

    low_carb = client.post("/api/assess", json={
        "labs": LABS_FULL, "meal": {**MEAL, "carbs": 20}, "pre_meal_glucose": 104}).json()
    ok(low_carb["exceeds_140"]["probability"] < worse["exceeds_140"]["probability"],
       f"{name}: less carbohydrate lowers the probability")

    # Personalization must be inert with no history — an app that "personalizes"
    # before it has evidence is inventing the thing it claims to learn.
    cold = client.post("/api/assess", json={"labs": LABS_FULL, "meal": MEAL,
                                            "history": []}).json()
    ok(cold["personalization"]["shrinkage"] == 0.0 and
       cold["personalization"]["offset_applied"] == 0.0,
       f"{name}: zero history applies zero personalisation")

    hist = [{"meal": MEAL, "observed_peak": 132, "observed_iauc": 55}] * 5
    warm = client.post("/api/assess", json={"labs": LABS_FULL, "meal": MEAL,
                                            "history": hist}).json()
    ok(abs(warm["personalization"]["shrinkage"] - 5 / 10) < 1e-6,
       f"{name}: shrinkage follows k/(k+5) exactly")

    # Every returned edit must actually lower risk.
    edits = alts["edits"]
    ok(all(e["delta_probability"] < 0 for e in edits),
       f"{name}: every suggested edit lowers the probability ({len(edits)} edits)")
    ok(all(e["probability"] <= alts["original"]["probability"] for e in edits),
       f"{name}: no edit is worse than the original meal")

    # Malformed input must be refused, not guessed at.
    bad = client.post("/api/assess", json={"labs": {}, "meal": {"carbs": 10}})
    ok(bad.status_code == 422, f"{name}: a meal missing meal_type is refused (422)")
    validate(bad.json(), "Error", f"{name} error body")

    # A typo'd 660 for 66 must be refused, not scored. A model will happily
    # return a confident number for an impossible meal.
    typo = client.post("/api/assess", json={"labs": LABS_FULL,
                                            "meal": {**MEAL, "carbs": 6600}})
    ok(typo.status_code == 422, f"{name}: an out-of-range macro is refused (422)")

    # ---- pillar 2: value of information -------------------------------------
    blind = client.post("/api/lab-value",
                        json={"labs": {}, "pre_meal_glucose": 104}).json()
    validate(blind, "LabValueResponse", f"{name} /api/lab-value")
    covered = client.post("/api/lab-value", json={
        "labs": {"a1c_pdl_lab": 5.6, "insulin": 6.0, "fasting_glu___pdl_lab": 95},
        "pre_meal_glucose": 104}).json()
    ok(blind["score"] == 1.0 and covered["score"] == 0.0,
       f"{name}: lab-value is 1.0 with no labs and 0.0 with a core panel on file")
    ok(covered["recommended_panel"] is None,
       f"{name}: no panel is recommended when a draw would not help")
    ok(blind["recommended_tier"] != "full",
       f"{name}: never asks for a full panel (lipids add ~0.002 AUC)")
    ok(blind["auc_after_draw"] > blind["auc_now"],
       f"{name}: the quoted gain is a real AUC gap")

    # ---- pillar 4: explanations ---------------------------------------------
    narration = client.post("/api/explain", json=body).json()
    validate(narration, "ExplainResponse", f"{name} /api/explain")
    ok(narration["source"] in ("claude", "template"),
       f"{name}: explanation declares its source ({narration['source']})")
    ok(len(narration["drivers_used"]) > 0,
       f"{name}: explanation carries the attributions it was built from")
    banned = ("dangerous", "diagnos", "you should eat")
    blob = (narration["headline"] + " ".join(narration["drivers"])
            + narration["caveat"]).lower()
    ok(not any(b in blob for b in banned),
       f"{name}: explanation contains no banned phrasing")

    # No returned meal may carry a field the contract does not define.
    allowed = set(SPEC["components"]["schemas"]["Meal"]["properties"])
    extra = {k for e in alts["edits"] for k in e.get("resulting_meal", {})} - allowed
    ok(not extra, f"{name}: edits echo no undefined meal fields (found {sorted(extra)})")


print("contract:", SPEC["info"]["title"], SPEC["info"]["version"])
from stub_server import app as stub_app  # noqa: E402

check(TestClient(stub_app), "STUB")

if "--real" in sys.argv:
    try:
        from serve import app as real_app

        check(TestClient(real_app), "REAL")
    except ImportError:
        print("\nserve.py not present yet — skipping the real backend")

print("\n" + "=" * 60)
if FAILURES:
    print(f"{len(FAILURES)} FAILED:")
    for msg in FAILURES:
        print("  -", msg)
    sys.exit(1)
print("all checks passed")
sys.exit(0)
