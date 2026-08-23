"""Pins every repair in cgm.py to a number measured from the published files.

A failure here means a repair regressed, not that a metric drifted. Exits
non-zero so CI can see it.
"""
import sys

import numpy as np
import pandas as pd

import cgm

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

FAILURES = []


def ok(cond, msg):
    if not cond:
        FAILURES.append(msg)
    print(("PASS  " if cond else "FAIL  ") + msg)


cgm.ls()

# ---------------------------------------------------------------- meal_type
m = cgm.meals()
print("\n-- meals", m.shape)
ok(len(m) == 1706, f"1,706 logged meals (got {len(m)})")
ok(m.subject.nunique() == 45, f"45 subjects (got {m.subject.nunique()})")
ok(set(m.meal_type.dropna()) == set(cgm.MEAL_TYPES),
   f"meal_type collapses to 4 categories: {sorted(set(m.meal_type.dropna()))}")
ok(m.meal_type_raw.nunique() == 10,
   f"10 raw strings preserved in meal_type_raw (got {m.meal_type_raw.nunique()})")
_counts = m.meal_type.value_counts().to_dict()
ok(_counts == {"dinner": 492, "breakfast": 436, "lunch": 435, "snack": 343},
   f"normalised counts match measurement: {_counts}")
ok(not any(c in m.columns for c in cgm.EMPTY_COLS),
   "near-empty columns dropped by default")
ok(all(c in cgm.meals(keep_empty=True).columns for c in cgm.EMPTY_COLS),
   "keep_empty=True retains them")

# casing is a per-subject habit -> raw string leaks subject identity
_raw_cap = m.groupby("subject").meal_type_raw.apply(lambda s: s.str[0].str.isupper().mean())
ok(int((_raw_cap == 1).sum()) == 4,
   f"4 subjects capitalise throughout (got {int((_raw_cap == 1).sum())})")
ok(int((_raw_cap == 0).sum()) == 27,
   f"27 subjects never capitalise (got {int((_raw_cap == 0).sum())})")

# ---------------------------------------------------------------- photos
mp = cgm.meals(with_photo=True)
print("\n-- meals with photo", mp.shape)
ok(int(mp.image_path.notna().sum()) == 1644,
   f"1,644 meals carry a photo (got {int(mp.image_path.notna().sum())})")
ok(int(mp.photo_ok.sum()) == 1644,
   f"every meal photo resolves into the zip (got {int(mp.photo_ok.sum())}/1644)")
ok(cgm.zip_member(1, "photos/00000005-PHOTO-2020-5-1-14-23-0.jpg")
   == "CGMacros-001/00000005-PHOTO-2020-5-1-14-23-0.jpg",
   "zip_member maps photos/<base> -> CGMacros-NNN/<base>")
ok(len(cgm.photo_members()) == 3454, f"3,454 photos in the zip (got {len(cgm.photo_members())})")

# ---------------------------------------------------------------- after photos
ap = cgm.after_photos()
print("\n-- after_photos", ap.shape)
ok(len(ap) == 1553, f"1,553 post-meal photos held separate (got {len(ap)})")
ok(ap.meal_type.isna().all(), "after-photos carry no meal_type")
ok(int(ap.calories.notna().sum()) == 0, "after-photos carry no macros at all")
ok(len(set(ap.image_path) & set(m.image_path.dropna())) == 0,
   "after-photos do not overlap the meals table")
ok(len(ap) + int(mp.image_path.notna().sum()) == 3197,
   "1,553 + 1,644 accounts for every photo row in the timeseries")
# the three unresolvable references are all after-photos, so no meal loses its image
ok(int(ap.photo_ok.sum()) == 1550,
   f"1,550 of 1,553 after-photos resolve (got {int(ap.photo_ok.sum())})")
_broken = ap[~ap.photo_ok]
ok(sorted(set(_broken.subject)) == [1, 2],
   f"the 3 broken references belong to subjects 1 and 2 (got {sorted(set(_broken.subject))})")

# ---------------------------------------------------------------- sensors
ts = cgm.timeseries()
print("\n-- timeseries", ts.shape)
ok(len(ts) == 687580, f"687,580 minute rows (got {len(ts)})")
_both = ts.dropna(subset=["libre_gl", "dexcom_gl"])
ok(len(_both) == 629605, f"629,605 both-sensor minutes (got {len(_both)})")
_off = (_both.dexcom_gl - _both.libre_gl).mean()
ok(abs(_off - 31.95) < 0.05, f"dexcom - libre = {_off:+.2f} mg/dL")
_per = _both.groupby("subject").apply(
    lambda g: (g.dexcom_gl - g.libre_gl).mean(), include_groups=False)
ok(int((_per < 0).sum()) == 1,
   f"exactly 1 subject has libre reading higher (got {int((_per < 0).sum())})")
ok(abs(100 * ts.libre_gl.notna().mean() - 100) < 0.05, "libre covers ~100% of rows")
ok(abs(100 * ts.dexcom_gl.notna().mean() - 91.6) < 0.1, "dexcom covers ~91.6%")
_sel = cgm.timeseries(sensor="dexcom_gl")
ok(_sel.glucose.equals(_sel.dexcom_gl), "sensor= exposes the chosen column as `glucose`")
try:
    cgm.timeseries(sensor="both")
    ok(False, "invalid sensor should raise")
except ValueError:
    ok(True, "invalid sensor raises ValueError")

# ---------------------------------------------------------------- subjects
b, g = cgm.bio(), cgm.gut_health()
print("\n-- bio", b.shape, "| gut", g.shape)
ok(len(b) == 45 and len(g) == 47, f"bio 45 rows, gut 47 (got {len(b)}, {len(g)})")
ok(sorted(set(range(1, 50)) - set(b.subject)) == [24, 25, 37, 40],
   "subject ids 1-49 missing 24/25/37/40")
ok(sorted(set(g.subject) - set(b.subject)) == [24, 25],
   "subjects 24 and 25 have gut panels but no CGM data")
ok(cgm.microbes().shape == (45, 1980), f"microbes is 45 x 1,980 (got {cgm.microbes().shape})")

# ---------------------------------------------------------------- Big Ideas
bw = cgm.bigideas_wearable()
bf = cgm.bigideas_food()
bc = cgm.bigideas_cgm()
print("\n-- bigideas wearable", bw.shape, "| food", bf.shape, "| cgm", bc.shape)
ok(int(bw.rmssd_valid.sum()) == 63994,
   f"n_beats>=30 leaves 63,994 usable HRV minutes (got {int(bw.rmssd_valid.sum())})")
ok(int((bw.n_beats == 0).sum()) == 90925,
   f"90,925 minutes have zero detected beats (got {int((bw.n_beats == 0).sum())})")
ok(int(cgm.bigideas_wearable(min_beats=10).rmssd_valid.sum()) == 87109,
   "min_beats is configurable (10 -> 87,109)")
ok(int(bf.schema_variant.sum()) == 58,
   f"58 rows flagged as the 003 schema variant (got {int(bf.schema_variant.sum())})")
ok(set(bf.loc[bf.schema_variant, "participant_id"]) == {"003"},
   "all of them belong to participant 003")
ok(int(bf.amount_num.isna().sum()) == 293,
   f"293 free-text amounts (got {int(bf.amount_num.isna().sum())})")
ok(bc.participant_id.nunique() == 16, "16 Big Ideas participants")

print("\n" + "=" * 60)
if FAILURES:
    print(f"{len(FAILURES)} FAILED:")
    for msg in FAILURES:
        print("  -", msg)
    sys.exit(1)
print("all checks passed")
sys.exit(0)
