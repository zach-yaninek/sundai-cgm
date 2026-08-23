# sundai-cgm

**Why does the same meal spike one person and not another?**

Built on the `cgm/` kit of the MIT Sundai Hack 137 data bucket. The project
started out predicting glucose response from a meal photograph, and the data
redirected it: CGMacros turns out to be largely a **standardized-meal study**,
so for most of it everyone ate the same food and the *person* is the variable.

An **iterative model ladder**, where each rung answers one question and the
previous rung is the number it has to beat:

| Rung | Input | Question | Cold-start R² |
|---|---|---|---|
| 0 | — | Floors: global mean, and a returning user's own average | — |
| 1 | Logged macros | What is knowable from a *perfect* food log? | 0.208 |
| 3 | CLIP image embedding | Does the photo carry signal the macros miss? | **0.026** |
| 4 | Meal identity + fasting labs | **Can a lab panel predict a stranger's response?** | **0.308** |

Rung 2 (photo → estimated macros → response) is written and runs against either
Claude or a local VLM, but was cut once the study structure became clear — see
*Why rung 2 was dropped*.

## Install

```bash
git clone https://github.com/zach-yaninek/sundai-cgm.git
cd sundai-cgm
pip install -r requirements.txt
python test_cgm.py
```

Data downloads on first use into `~/.cgm_cache` (`$CGM_CACHE` to override), with
the User-Agent header the bucket requires — `r2.dev` returns 403 to Python's
default.

## Use

```python
import cgm, targets

cgm.ls()                              # tables + cache status
m  = cgm.meals(with_photo=True)       # 1,706 meals, meal_type normalised
ap = cgm.after_photos()               # 1,553 post-meal shots, held separate
ts = cgm.timeseries(sensor="libre_gl")

d  = targets.modelling_set()          # 1,382 rows — prints its own funnel
c  = targets.curve(subject=1, timestamp=d.timestamp.iloc[0])
```

Reproduce the ladder:

```bash
python rung1_macros.py      # logged macros -> response
python embed.py             # CLIP vectors, ~18s on Apple MPS
python rung3_clip.py        # image embedding -> response
python rung4_subject.py     # meal + labs -> response   (the headline)
```

## Results

Target is **iAUC over 120 minutes** (mg/dL·h) — the area between the glucose
curve and its own starting value. Cross-validation is grouped by subject, and
**cold-start** means the held-out person was never seen in training. That is the
only regime in which a claim about a *new* person is honest.

### Rung 4 — the headline

On the 857 standardized meals (16 dishes, 45 subjects):

| Model | n | MAE | 95% CI | R² |
|---|---|---|---|---|
| Global mean *(floor)* | 857 | 40.20 | [35.8, 45.2] | −0.016 |
| Meal identity only | 857 | 35.14 | [30.7, 40.4] | 0.178 |
| **Meal + fasting labs** | **857** | **32.56** | **[28.5, 37.0]** | **0.308** |
| Meal + labs + gut panel | 857 | 33.03 | [28.7, 37.7] | 0.290 |
| Labs only, no meal | 857 | 37.82 | [32.9, 43.0] | 0.053 |

**Knowing a stranger's fasting labs improves prediction over knowing only what
they ate** — R² 0.308 against meal identity's 0.178. The strongest single
predictors are fasting glucose, HOMA-IR and HbA1c, which is the result a
clinician would expect and is reassuring rather than novel.

The **gut panel does not help cold-start** (0.290 vs 0.308). 22 Viome features
across 45 subjects is over-parameterised, and what looks like a gain in-sample
is fingerprinting. The headline model omits it.

### Rungs 0–3

| Model | Regime | n | MAE | 95% CI | R² |
|---|---|---|---|---|---|
| Global mean *(floor)* | cold | 1,382 | 36.24 | [32.5, 40.4] | −0.020 |
| Carbs only | cold | 1,382 | 33.43 | [29.9, 37.6] | 0.112 |
| **Rung 1 — all macros** | cold | 1,382 | 31.13 | [27.9, 34.7] | 0.208 |
| Rung 3 — CLIP PCA-64 | cold | 1,382 | 34.75 | [31.2, 38.7] | **0.026** |
| Rung 3 — CLIP + macros | cold | 1,382 | 31.38 | [28.1, 34.9] | 0.214 |
| Subject mean *(floor)* | known | 1,382 | 30.76 | [26.3, 35.2] | 0.210 |
| Rung 1 — all macros | known | 1,382 | 29.63 | [26.6, 32.8] | 0.267 |

Two things worth reading off this table. **A generic image embedding carries
almost nothing** — R² 0.026 cold-start, barely distinguishable from predicting
the mean. And **for a returning user, their own historical average (30.76) very
nearly matches the full macro model (29.63)**, intervals overlapping: knowing
*who* ate the meal is about as informative as knowing *what* they ate.

### What the data turned out to be

| | |
|---|---|
| Meals whose macro combination recurs 20+ times | **857 of 1,382 (62%)** |
| Distinct standardized dishes | **16** |
| Subjects eating each top dish | **43–44 of 45** |
| CLIP cosine within a dish vs across | **0.714 vs 0.559** |
| iAUC range for one identical meal (66 g carb, 712 kcal) | **6 to 253** |
| Per-subject mean iAUC | **13 to 112 (8.7×)** |

Variance in glucose response, on the standardized subset:

| Explained by | |
|---|---|
| **Subject identity** | **28.6%** |
| Meal identity | 20.8% |

### Two caveats that limit these numbers

**The `known` regime inflates rung 4 and must not be quoted.** Every subject has
a unique lab vector — BMI alone separates all 45 — so when the same person
appears in training the lab panel acts as a subject ID. With 45 subjects × 16
dishes = 720 cells and 857 rows, the model can nearly memorise each person-meal
pair, which is why it appears to beat the subject-identity oracle there. The
cold-start rows are the claim.

**The effective n is 45, not 857.** Lab features vary only between people, so
the personalisation mapping is learned from 45 subjects however many meals they
each ate. The confidence intervals are bootstrapped over subjects for that
reason. This is enough to demonstrate; it is not enough to claim.

## The app

```bash
uvicorn serve:app                      # terminal 1
cd web && npm install && npm run dev    # terminal 2 -> localhost:5173
```

Three tabs, one per pillar:

- **Assess a meal** — macros in, calibrated probability of exceeding 140 mg/dL,
  predicted peak and curve, then the changes that would lower it.
- **Would a test help?** — value of information. Answers whether drawing bloods
  would sharpen the prediction, and asks for the three analytes that carry the
  signal rather than a full panel.
- **Learning** — the measured learning curve, and a log of what actually happened
  after meals you assessed.

Under the numbers on the first tab, an **explanation** narrates what drove the
estimate, built from the model's own SHAP attributions. It labels itself
`written by Claude` or `generated locally` — the local template is the default
path when no `ANTHROPIC_API_KEY` is set, and it is a supported path rather than a
degraded one.

Deployment: `Dockerfile` + `render.yaml` for the API, `web/` on Vercel with
`VITE_API_BASE` pointing at it. The serving image is 5.6 MB of code and
artifacts — it deliberately excludes the training modules, which is why the
shrinkage maths lives in `shrinkage.py` rather than `personalize.py`.

## Serving

`predict.py` is the whole serving surface. It reads **only** from `artifacts/` —
no training code path, no network, no source data — so a deployment is a small
container reading a 30 KB booster.

```python
import predict

predict.rank_meals(labs)                       # every dish, best-tolerated first
predict.predict_response(labs, "66.0/712.0")   # one dish, with a curve to plot
predict.dish_catalog()                         # the 16 dishes and their spread
predict.required_labs()                        # which fields the model uses
```

The product question it answers: **given this person's bloodwork, which of these
meals is gentlest for them?** Two illustrative panels across the same 16 dishes:

| Panel | HOMA-IR | Predicted iAUC range |
|---|---|---|
| Metabolically healthy | 0.9 | **0 – 64** |
| Insulin resistant | 5.1 | **24 – 133** |

Design decisions worth knowing:

- **Column order is pinned in `feature_spec.json`, not in convention.** The
  serving vector is rebuilt from the recorded order, because training/serving
  skew is invisible when it happens — the model still returns a confident number.
- **A partial panel degrades rather than errors.** Missing labs fall back to
  training medians, and `imputed_fields` reports which, so a prediction resting
  on six guesses is visibly weaker than one that isn't.
- **`homa_ir` is derived** from glucose × insulin / 405 when both are present,
  so callers don't need the formula.
- **Predictions clip at zero.** iAUC is the area *above* baseline and cannot be
  negative. The reported evaluation metrics do *not* include this clip, so they
  are marginally conservative rather than flattered by it.
- **Unknown dishes and unknown lab fields raise.** A silently ignored typo in a
  field name is a wrong prediction, not a missing one.

`test_predict.py` reloads the booster through the public API and checks it
reproduces training-time predictions row for row — currently exact, max
difference 0.00e+00. That test is what stands between this and a demo that
looks fine and is wrong.

## Why rung 2 was dropped

`annotate.py` is complete and runs against Claude (Batches API) or a local VLM
(Ollama), writing an identical `vision_macros.parquet` either way. It was cut on
evidence, not for time:

- With **16 dishes covering 62% of the data**, estimating macros from a photo is
  a lookup problem, not an estimation problem — and the macros barely vary.
- A local `qwen2.5vl:7b` ran **12.1 s/image** and, on a 12-photo probe, its carb
  estimates ranked *inversely* to the logged values (Spearman −0.706, median
  ratio 0.49). It systematically under-read portions.
- Rung 3 had already shown the image contributes R² 0.026 cold-start.

The code is kept because the finding is worth reproducing, and because the
provider contract is the part that would matter if this ran on a dataset where
the meals actually differed.

## What the loader repairs

Every defect below was measured from the published files, and every one is
pinned by an assertion in `test_cgm.py`.

| Defect | Repair |
|---|---|
| `meal_type` has **10 strings for 4 categories**, and its casing is a per-subject habit that leaks subject identity | Normalised to 4; raw kept in `meal_type_raw` |
| Photo paths don't resolve — parquet says `photos/<base>`, zip says `CGMacros-<subject:03d>/<base>` | Mapped on subject; **all 1,644 meal photos resolve** |
| 3,197 photo rows but only 1,706 have macros | The other **1,553 are post-meal shots** (median 16 min later) — `after_photos()`, never auto-joined |
| Dexcom reads **+31.95 mg/dL** above Libre; per-subject −24.3 to +71.7 | Default `libre_gl`; explicit `sensor=`; never averaged |
| `recordindex` 0.00% populated, `sugar` 0.01%, `steps` 0.82% | Dropped unless `keep_empty=True` |
| Big Ideas `rmssd_ms` — median minute has **3 detected beats**, 90,925 have zero | `rmssd_valid` flags `n_beats >= 30` (63,994 minutes) |
| Big Ideas participant `003` has a different food-log schema | 58 rows flagged `schema_variant`, unmatched columns kept as `unmapped_a/b` |
| Big Ideas `amount` is free text ("half", "rest of bar") | 293 of 1,422 — use `amount_num` |

Three photo references in the published data are broken. All three are
after-photos, so the modelling set loses nothing.

## The honest n

`targets.modelling_set()` prints this every run rather than letting anyone quote
the headline:

```
logged meals                            1,706
window >= 80% covered                   1,685  (-21)
complete macros                         1,684  (-1)
photo resolves                          1,623  (-61)
no second meal within 120min            1,382  (-241)
subjects                                   45
```

`peak_delta` and `iauc` correlate 0.953, so they are near interchangeable;
`iauc` is primary because it is less sensitive to one noisy reading.

## Version pin you cannot ignore

**`xgboost>=3.1` is a hard floor, and it forces Python >= 3.10.**

xgboost 3.1+ serialises the learned `base_score` as an array. Older versions
cannot parse that, fall back to the default 0.5, and return predictions roughly
**53 mg/dL·h too low — with no error raised**. CI caught this on the Python 3.9
job, where pip resolves xgboost to 2.1.4.

`predict.py` now reads `base_score` back from the loaded booster and compares it
against the value recorded in `rung4_feature_spec.json`, so a version that would
mis-read the model raises `ArtifactError` instead of serving confident nonsense.

## Data licences

**This repository does not redistribute the data or the photographs.**

- **CGMacros** — Gutierrez-Osuna et al., PhysioNet 2025, `doi:10.13026/3z8q-x658`.
  **CC BY-NC-SA 4.0 — non-commercial and share-alike.** Fine for a hackathon
  demo. A deployed app built on it must stay non-commercial, and share-alike
  attaches to any derived database you publish.
- **Big Ideas Lab** — Cho et al., PhysioNet, `doi:10.13026/zthx-5212`.
  **ODC-BY 1.0** — attribution only, commercial use fine.

Cite the original authors, not this repository or the bucket. Code is MIT; see
`LICENSE`.
