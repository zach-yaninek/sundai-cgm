# Frontend brief

Everything you need to build the app without waiting on the model.

## Start here

```bash
uv run --with fastapi --with uvicorn python contract/stub_server.py
# → http://127.0.0.1:8000   (docs at /docs)
```

That's a **fixture server**: every endpoint returns the real response shape with
invented numbers. No model, no data files. Build the whole frontend against it.
When `serve.py` lands it implements the identical shapes and you change one base
URL. `test_contract.py` validates both against the same `openapi.json`, so a
screen built here keeps working after the swap.

The fixtures **respond to their input** — worse labs and more carbohydrate raise
the probability, missing fields widen the confidence band — so you can reach every
visual state without a model. None of the numbers are predictions.

Your app lives entirely in `web/`. Nothing you write touches Python.

## The contract

`contract/openapi.json` is the source of truth. Four endpoints:

| | |
|---|---|
| `GET /api/meta` | Cohort size, thresholds, performance, disclaimer text, exclusion list. **Call once at startup and render from it** — don't hardcode numbers. |
| `GET /api/fields` | The lab form: names, units, ranges, cohort medians, `importance_rank`. Order the form by that rank. |
| `POST /api/assess` | The risk call for one meal. |
| `POST /api/alternatives` | Suggested changes to that meal. |

The server is **stateless**. The lab panel and the meal history live in browser
localStorage and get sent on each request. No personal health data reaches the
server — that's the privacy story, so please don't add a backend session.

## Screens

**Onboarding** — lab form from `/api/fields`. Every field optional; missing ones
fall back to cohort medians and come back in `confidence.imputed_fields`. Show
that filling more fields narrows the estimate. Gate on the disclaimer and the
exclusion list from `/api/meta`.

**Meal input** — macros, meal type, optional pre-meal glucose. Validate hard:
carbohydrate in the source data runs 0–761 g, and a typo'd 660 instead of 66 must
not sail through. Pre-meal glucose is optional but worth prompting for — it lifts
flag AUC from 0.841 to 0.888.

**Risk card** — the probability, the cohort band, predicted peak, the curve.
`confidence.band` is `narrow` / `moderate` / `wide`; **a wide band has to look
wide.** If the UI shows 72% in the same confident type either way, the honesty in
the API is wasted.

**Alternatives** — `edits[]`, each with its `delta_probability`. The API returns
them gentlest-ask-first (that is what `EDIT_LADDER` encodes); the shipped UI
re-sorts to most-effective-first, on the grounds that 67%, 67%, 73% reads as
unsorted. `from_your_history[]` is the user's own past low-response meals and is
empty until they've logged some. Empty `edits` means the meal is already in their
lower range — say that, don't invent a suggestion.

**History** — localStorage log of meals and observed outcomes, feeding the
`history` field. Show `personalization.meals_logged` climbing.

**Learning curve** — plot `meta.performance.learning_curve`. Nine points, real,
measured on held-out people. Cheap to build and it's the most convincing thing in
the app. It bends at 6 logged meals, where the correction stops being a flat
offset and gains a slope; `personalization.learned_slope` says which side of that
a given response is on.

## Copy rules — these are not style preferences

- **Never the word "dangerous."** State the modelled probability against a named
  threshold: *"72% likely to exceed 140 mg/dL — higher than 80% of this cohort's
  meals."*
- **Alternatives are predictions, not advice.** "Changes this model predicts would
  lower your response" is defensible. "Eat this instead" is dietary advice.
- **n = 45 belongs on the results screen**, not only in an About page.
- The exclusion list from `/api/meta` is a **gate at onboarding**, not fine print.
  Type 1 diabetes, insulin dosing, glucose-lowering medication, pregnancy, under 18.

## Numbers that must appear in the UI

| | |
|---|---|
| Flag AUC | **0.888** with pre-meal glucose, **0.841** without |
| Cohort | **45 subjects**, 1,382 meals, ages 24–59, one study |
| Personalisation | **~21% better after 15 logged meals** (MAE 28.9 → 22.7) |
| Threshold | 140 mg/dL, which **45.8%** of cohort meals exceeded |

All of these come from `/api/meta` — read them from there so they stay true when
the real model replaces the stub.

## Checking your work

```bash
uv run --with fastapi --with httpx --with jsonschema python test_contract.py
```

13 checks: schema validation on every endpoint, all three confidence bands
reachable, healthier panels and lower-carb meals reduce the probability, zero
history applies zero personalisation, and malformed input returns 422 rather than
a guess.
