# web/ — working notes

The app is built. These are the rules that keep it honest, and the traps worth
knowing before changing anything.

Three pillars, one per tab: **assess a meal**, **would a blood test help**, and
**learning**. A fourth thing — the explanation panel — sits under the numbers on
the first tab.

## Run it

```bash
# terminal 1 — the real backend
uvicorn serve:app
# ...or the fixture server, if you only want to touch UI:
# uv run --with fastapi --with uvicorn python contract/stub_server.py

# terminal 2 — this app
cd web && npm install && npm run dev     # http://localhost:5173
```

Vite proxies `/api` to `127.0.0.1:8000`, so the browser sees one origin and CORS
never comes up. `npm run dev` against a dead backend shows a clear message
telling you to start it — that is intentional, not a bug.

In a deployed build there is no proxy, so `VITE_API_BASE` supplies the API's
absolute origin instead. See `web/.env.example`.

## The rule that matters most

**`src/api/schema.d.ts` is generated. Never edit it, and never hand-write types
that duplicate it.**

```bash
npm run gen:api      # regenerate from ../contract/openapi.json
npm run check:api    # fails if the checked-in file is stale
```

Every type in `src/api/client.ts` is derived from that file. When the backend
changes a response shape, they change `contract/openapi.json`, you regenerate,
and TypeScript points at every call site that needs updating. That is the entire
mechanism keeping the two halves in sync. If you define your own `interface
AssessResponse` anywhere, you have opted out of it and drift becomes silent until
the demo.

`npm run build` regenerates before compiling, so a stale schema cannot ship.

**If you need a field the API doesn't return, do not invent it client-side.** Add
it to the contract instead. A number computed in the frontend that looks like a
model output is the worst failure mode this project has.

## What to build

Six screens. `src/components/` has a stub for each with a TODO.

| Screen | Notes |
|---|---|
| **Onboarding** | Build the form from `getFields()`. Every field is optional — order by `importance_rank` so someone filling only the first three gets most of the available accuracy. Show which fields were imputed and that filling more narrows the estimate. Gate on `meta.disclaimer` and `meta.exclusions`. |
| **Meal input** | Macros, meal type, optional pre-meal glucose. **Validate hard** — carbohydrate in the source data runs 0–761 g, and a typo'd 660 for 66 must not sail through. Prompt for pre-meal glucose: it lifts flag AUC from 0.836 to 0.885. |
| **Risk card** | Probability, cohort band, predicted peak, curve. Use `describeRisk()` from the client for the wording. |
| **Alternatives** | `edits[]`, smallest effective change first, each with `delta_probability`. `from_your_history[]` is the user's own past low-response meals — empty until they log some. **Empty `edits` means the meal is already in their lower range: say that, don't invent a suggestion.** |
| **History** | `src/lib/storage.ts` already handles persistence. Log the *observed* outcome the user enters later, never the prediction. Show `personalization.meals_logged` climbing. |
| **Learning curve** | Plot `meta.performance.learning_curve`. Seven real points measured on held-out people. Cheapest thing in the app and the most convincing. |
| **Would a test help?** | Value of information. **Never tell a CGM wearer to get a glucose panel** — they already have continuous glucose. Lead with the analytes a draw actually adds: HbA1c and fasting insulin. Score 0 means say so and offer nothing. |
| **Explanation** | Lazy-loaded under the numbers, so an external call can never delay the assessment. Show `source` — hiding whether Claude or the local template wrote it would misrepresent the system. |

`Chart.tsx` is a hand-rolled SVG line chart used by both curves. No charting
library: two series of under 30 points do not justify ~500 KB, and drawing it
ourselves is what lets a wide confidence band render as a dashed, faded line.

**`zeroBaseline` matters.** The glucose curve anchors at zero; the learning curve
must not. Forcing zero there squashes 28.7 → 23.4 into a flat line and hides the
entire effect the chart exists to show.

## Rules that are not style preferences

**Never render numbers you hardcoded.** `n = 45`, AUC 0.885, the threshold, the
disclaimer text — all come from `getMeta()`. They change when the model is
retrained, and a hardcoded one becomes a false claim on stage.

**Never the word "dangerous."** State the modelled probability against a named
threshold: *"72% likely to exceed 140 mg/dL — higher than 80% of this cohort's
meals."* `describeRisk()` does this for you; use it rather than composing your own
sentence.

**Alternatives are predictions, not advice.** "Changes this model predicts would
lower your response" is defensible. "Eat this instead" is dietary advice, and this
is a model fitted to 45 people. Do not let the copy drift into the second form
while you're moving fast — this is the sharpest edge in the product.

**A wide confidence band must LOOK wide.** `confidence.band` is
`narrow | moderate | wide`. `bandWeight()` gives you label and hint text. If 72%
renders in identical confident type either way, the honesty in the API is wasted.

**`n = 45` belongs on the results screen**, not only in an About page.

**The exclusion list is a gate at onboarding**, not fine print: type 1 diabetes,
insulin dosing, glucose-lowering medication, pregnancy, under 18. Someone who
selects one should not get a reading.

**Nothing personal leaves the browser.** The backend is stateless by design —
labs and history are sent per request and never stored server-side. Do not add a
session, a user table, or an analytics call carrying lab values or meals. Please
do build a "delete my data" control; `clearEverything()` is already there.

## Conventions

- TypeScript strict is on, including `noUncheckedIndexedAccess`. `arr[0]` is
  `T | undefined` — that is deliberate.
- Generated schema types make most fields optional, because OpenAPI marks few of
  them required. Use `?.` and provide fallbacks rather than asserting with `!`.
  `App.tsx` uses `!` in two places for brevity; do better in real components.
- `npm run typecheck` must pass before you commit.
- Keep components presentational. Fetching lives in `src/api/client.ts`,
  persistence in `src/lib/storage.ts`.

## Changing a response shape

Change `contract/openapi.json` first, then `npm run gen:api`, and the compiler
shows you every call site that broke. `test_contract.py --real` validates the
stub *and* `serve.py` against that one file, and CI runs `check:api` so a stale
generated type fails the build rather than surfacing at demo time.
