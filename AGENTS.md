# Notes for coding agents

Read this before touching the code or fetching data. Full detail is in
`README.md`; this is the short version of what wastes time.

## Getting the data

Data is **not in this repo** and must not be committed. It downloads on first use:

```python
import cgm
m = cgm.meals(with_photo=True)      # fetches + caches to ~/.cgm_cache
```

Set `CGM_CACHE` to relocate. `cgm.ls()` shows what is cached.

**If you fetch from the bucket directly, send a browser User-Agent.** The host
(`r2.dev`) returns **HTTP 403** to Python's default agent, so a bare
`pd.read_parquet(url)` fails. This is the most common way to get stuck here.

```python
pd.read_parquet(URL, storage_options={"User-Agent": "Mozilla/5.0"})
```

`artifacts/` **is** committed — it is what a webapp reads. The `.gitignore`
ignores `*.parquet` broadly and re-includes `artifacts/*.parquet`. If you add an
artifact file type, check it is not silently ignored.

## Traps specific to this dataset

- **Never pool `libre_gl` and `dexcom_gl`.** Dexcom reads +31.95 mg/dL above
  Libre on average, per-subject offsets run −24.3 to +71.7, and a model trained
  on the pooled column mostly learns which sensor was worn. Default is
  `libre_gl` (100% coverage vs 91.6%). Pass `sensor=` and say which you chose.
- **`meal_type` is normalised for you.** The raw column has ten strings for four
  categories and its casing is a per-subject habit, so `meal_type_raw` partly
  encodes subject identity. Do not feature-engineer on `meal_type_raw`.
- **`after_photos()` is separate on purpose.** 1,553 of the 3,197 photos are
  post-meal shots with no macros. Joining every `image_path` row to the meal
  table gives you 1,553 rows with a NaN target.
- **Split by `subject`, never at random** — use `evaluate.py`. Meals within a
  person are correlated.
- **Know which regime you are quoting.** `cold` holds out whole subjects (the
  webapp case); `known` puts some of a subject's meals in training. A
  per-subject-mean baseline only exists in `known`. Mixing them up turns a weak
  result into a false claim.

## Working on this code

- **Requires pandas ≥ 2.2**, tested on Python 3.9–3.13. `lightgbm` is not a
  dependency — use `xgboost`.
- **Run `python test_cgm.py` after any change.** It exits non-zero and is what
  CI runs. Every assertion is pinned to a number measured from the published
  files, so a failure means a repair regressed.
- **Do not "simplify" the repairs in `cgm.py`.** Each exists because the
  published file is wrong in a specific way; the reasoning is in the `NOTES`
  block at the bottom of that file. Read it first.
- `targets.modelling_set()` prints its funnel every run. The honest n is
  **1,382**, not the 1,706 headline. Quote the printed number.

## The API layer

`contract/openapi.json` is the frozen contract between the Python side and the
React app in `web/`. `contract/stub_server.py` implements it with fixtures so the
frontend can be built before the model exists; `serve.py` will implement the same
shapes for real.

**If you change a response shape, change `openapi.json` first.** `test_contract.py`
validates the stub and the real backend against that one file — that is the only
thing stopping the frontend being built against a shape the backend never returns.

The server is stateless by design: the lab panel and meal history live in browser
localStorage and are sent per request. Do not add server-side storage of personal
health data.

## Licences

Code is MIT. **CGMacros is CC BY-NC-SA 4.0** — non-commercial, share-alike;
a deployed app must stay non-commercial. **Big Ideas Lab is ODC-BY 1.0** —
attribution only, commercial use fine. Cite the original authors, not this repo.
