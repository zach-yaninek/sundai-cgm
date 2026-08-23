# sundai-cgm

Predicting the glucose response to a meal from its photograph — built on the
`cgm/` kit of the MIT Sundai Hack 137 data bucket.

An **iterative model ladder**, where each rung answers one question and the
previous rung is the number it has to beat:

| Rung | Input | Question |
|---|---|---|
| 0 | — | Floors: the global mean, and a returning user's own average |
| 1 | Logged macros | What is knowable from a *perfect* food log? |
| 3 | CLIP image embedding | Does the photo carry signal the macros miss? |
| 2 | Photo → VLM → estimated macros | What is a vision stage actually worth? |
| 4 | + labs, gut panel | Does personalisation beat knowing the person's average? |

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

## Results

Target is **iAUC over 120 minutes** (mg/dL·h) — the area between the glucose
curve and its own starting value. Cross-validation is grouped by subject.

| Model | Regime | n | MAE | 95% CI | R² |
|---|---|---|---|---|---|
| Global mean *(floor)* | cold | 1,382 | 36.24 | [32.5, 40.4] | −0.020 |
| Carbs only | cold | 1,382 | 33.43 | [29.9, 37.6] | 0.112 |
| **Rung 1 — all macros** | **cold** | **1,382** | **31.13** | **[27.9, 34.7]** | **0.208** |
| Global mean *(floor)* | known | 1,382 | 35.91 | [32.2, 40.1] | −0.004 |
| Subject mean *(floor)* | known | 1,382 | 30.76 | [26.3, 35.2] | 0.210 |
| Carbs only | known | 1,382 | 33.06 | [29.6, 37.1] | 0.136 |
| **Rung 1 — all macros** | **known** | **1,382** | **29.63** | **[26.6, 32.8]** | **0.267** |

**Two regimes, and they answer different questions.** `cold` holds out whole
subjects — a stranger uploads a photo, which is the webapp case. `known` puts
some of that person's other meals in training. A per-subject-mean baseline only
*exists* in `known`, because under cold folds the held-out subject has no
training rows at all.

**The most interesting number here is a floor, not a model.** For a returning
user, predicting their own historical average gives MAE 30.76 — and the full
macro model only reaches 29.63, with heavily overlapping intervals. Knowing
*who* ate the meal is very nearly as informative as knowing *what* they ate.
Carbohydrate alone (33.06) does not even match the personal average.

That is the bar the photo rungs have to clear, and it is a harder one than the
usual "beat the global mean" framing would suggest.

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
