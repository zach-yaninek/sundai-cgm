"""cgm.py — clean loaders for the cgm/ kit (MIT Sundai Hack 137).

    import cgm
    cgm.ls()                        # what's available
    m   = cgm.meals()               # 1,706 logged meals, meal_type normalised
    ts  = cgm.timeseries()          # 687,580 minute rows, empty columns dropped
    p   = cgm.photo_path(row)       # -> Path to the meal photo on disk

Files are fetched from the hack bucket on first use and cached in ~/.cgm_cache
(override with $CGM_CACHE).

Two studies live here. **CGMacros** is 45 people wearing two CGMs at once with
photographed, macro-annotated meals. **Big Ideas Lab** is 16 pre-diabetics with a
Dexcom and a research-grade wrist sensor.

Every loader returns a frame that is safe to group by its subject key. The known
defects in the published data are repaired or flagged here rather than left for
each notebook to rediscover; see NOTES at the bottom for what was changed and why.
"""
from __future__ import annotations

import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://pub-db539ca4ecb344ac9c7e32102b0908d1.r2.dev/cgm"
CACHE = Path(os.environ.get("CGM_CACHE", Path.home() / ".cgm_cache"))
UA = "Mozilla/5.0 (sundai-hack)"  # r2.dev 403s urllib's default UA

PHOTO_ZIP = "cgmacros_meal_photos_768px.zip"

TABLES = {
    "cgmacros_timeseries":    "minute-level CGM + Fitbit + meal rows (687,580 x 19)",
    "cgmacros_meals":         "just the rows where a meal was logged (1,706 x 19)",
    "cgmacros_bio":           "demographics + fasting labs + fingersticks (45 x 24)",
    "cgmacros_gut_health":    "Viome pathway scores, 1=not optimal 3=good (47 x 23)",
    "cgmacros_microbes":      "binary presence/absence per taxon (45 x 1,980)",
    "bigideas_cgm":           "Dexcom EGV readings, 5-minute cadence (36,898 x 3)",
    "bigideas_wearable_1min": "per-minute HR, RMSSD, EDA, skin temp (209,464 x 7)",
    "bigideas_food_log":      "meals with calories and macros (1,422 x 18)",
    "bigideas_participants":  "sex and HbA1c (16 x 3)",
}

# The four real meal categories. The published file uses ten strings for them --
# 'dinner'/'Dinner', 'snack'/'Snack'/'Snacks', and a lone 'snack 1'. Casing is a
# per-subject habit, so the raw string leaks subject identity; see NOTES.
MEAL_TYPES = ("breakfast", "lunch", "dinner", "snack")

# Columns that are effectively empty in the published timeseries: recordindex is
# 0.00% populated, sugar 0.01%, steps 0.82%. Dropped unless keep_empty=True.
EMPTY_COLS = ("recordindex", "sugar", "steps")

# Sensor disagreement is large and systematic -- Dexcom reads +31.95 mg/dL above
# Libre on average, and per-subject offsets run -24.3 to +71.7. Never pool them.
SENSORS = ("libre_gl", "dexcom_gl")
DEFAULT_SENSOR = "libre_gl"  # 100% coverage vs Dexcom's 91.6%

# Big Ideas RMSSD is computed per minute from however many beats were detected.
# The median minute has 3, and 90,925 minutes have none at all.
MIN_BEATS = 30


# ---------------------------------------------------------------- plumbing

def _fetch(filename: str) -> Path:
    """Download ``filename`` into the cache if it isn't there yet."""
    CACHE.mkdir(parents=True, exist_ok=True)
    local = CACHE / filename
    if not local.exists():
        req = urllib.request.Request(f"{BASE}/{filename}", headers={"User-Agent": UA})
        tmp = local.with_suffix(local.suffix + ".part")
        with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.rename(local)
    return local


def _raw(name: str) -> pd.DataFrame:
    return pd.read_parquet(_fetch(f"{name}.parquet"))


def ls() -> None:
    """Print the tables and whether each is cached locally."""
    print(f"cgm/ — CGMacros (45 people) + Big Ideas Lab (16), {BASE}")
    print(f"cache: {CACHE}\n")
    for name, desc in TABLES.items():
        mark = "cached" if (CACHE / f"{name}.parquet").exists() else "remote"
        print(f"  {name:<24} [{mark}]  {desc}")
    zp = CACHE / PHOTO_ZIP
    print(f"  {'meal photos (zip)':<24} [{'cached' if zp.exists() else 'remote'}]  "
          f"3,454 jpgs, 213 MB")


# ------------------------------------------------------- shared repairs

def normalise_meal_type(s: pd.Series) -> pd.Series:
    """Collapse the ten published strings onto the four real categories."""
    out = s.astype("string").str.strip().str.lower()
    out = out.str.replace(r"^snacks?(\s+\d+)?$", "snack", regex=True)
    return out


def _prepare(df: pd.DataFrame, *, keep_empty: bool) -> pd.DataFrame:
    """Repairs common to the timeseries and meals tables."""
    df = df.copy()
    if "meal_type" in df.columns:
        df["meal_type_raw"] = df["meal_type"]
        df["meal_type"] = normalise_meal_type(df["meal_type"])
    if not keep_empty:
        df = df.drop(columns=[c for c in EMPTY_COLS if c in df.columns])
    return df


def zip_member(subject, image_path: str) -> str:
    """Map a parquet ``image_path`` onto its entry in the photo zip.

    The parquet stores ``photos/<basename>`` while the zip is laid out as
    ``CGMacros-<subject:03d>/<basename>``. Resolving one to the other needs the
    subject id, which is why this is not a plain string operation.
    """
    return f"CGMacros-{int(subject):03d}/{str(image_path).split('/')[-1]}"


# ---------------------------------------------------------------- CGMacros

def timeseries(*, keep_empty: bool = False, sensor: str | None = None) -> pd.DataFrame:
    """Minute-level CGM, Fitbit and meal rows for all 45 CGMacros subjects.

    ``meal_type`` is normalised to the four real categories, with the published
    string preserved in ``meal_type_raw``. Near-empty columns are dropped unless
    ``keep_empty``.

    ``sensor`` selects a single glucose column and exposes it as ``glucose``.
    Both raw columns are always retained. Pooling ``libre_gl`` and ``dexcom_gl``
    into one column trains a model to identify the sensor, not the person.
    """
    df = _prepare(_raw("cgmacros_timeseries"), keep_empty=keep_empty)
    if sensor is not None:
        if sensor not in SENSORS:
            raise ValueError(f"sensor must be one of {SENSORS}, got {sensor!r}")
        df["glucose"] = df[sensor]
        df.attrs["sensor"] = sensor
    return df.sort_values(["subject", "timestamp"]).reset_index(drop=True)


def meals(*, keep_empty: bool = False, with_photo: bool = False) -> pd.DataFrame:
    """The 1,706 logged meals, one row each, with macros.

    ``amount_consumed`` is present but should not be used as a numeric feature
    without per-subject rescaling: it mixes percentages, small counts and values
    up to 900, and the convention varies by subject. See NOTES item 10.

    ``with_photo`` adds ``zip_member`` and ``photo_ok``. 1,644 meals carry a
    photo and **all 1,644 resolve** -- the three broken references in the
    published data are all post-meal shots, so no meal loses its image. See
    :func:`after_photos`.
    """
    df = _prepare(_raw("cgmacros_meals"), keep_empty=keep_empty)
    df = df.sort_values(["subject", "timestamp"]).reset_index(drop=True)
    if with_photo:
        df = _attach_photo(df)
    return df


def after_photos(*, keep_empty: bool = False) -> pd.DataFrame:
    """The 1,553 photos that carry no macros — post-meal 'what was left' shots.

    These sit in the timeseries alongside the logged meals and have no
    ``meal_type`` and no macros of any kind. They follow a logged meal by a
    median of 16 minutes, and 87% fall within an hour of one.

    They are deliberately not part of :func:`meals`. Joining every row with an
    ``image_path`` to the meal table yields 1,553 rows with a NaN target.

    1,550 of the 1,553 resolve into the zip. The 3 that do not are the only
    broken photo references in the dataset, and it is a small mercy that they
    landed here rather than in the meal set.
    """
    ts = _prepare(_raw("cgmacros_timeseries"), keep_empty=keep_empty)
    df = ts[ts["image_path"].notna() & ts["meal_type"].isna()].copy()
    df = df.sort_values(["subject", "timestamp"]).reset_index(drop=True)
    return _attach_photo(df)


def _attach_photo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    has = df["image_path"].notna()
    df["zip_member"] = pd.Series(
        [zip_member(s, p) if ok else None
         for s, p, ok in zip(df["subject"], df["image_path"], has)],
        index=df.index, dtype="object",
    )
    members = photo_members()
    df["photo_ok"] = df["zip_member"].isin(members)
    return df


def bio() -> pd.DataFrame:
    """Demographics, fasting labs and fingersticks, one row per subject.

    45 rows. Subject ids are not contiguous: they run 1-49 with 24, 25, 37 and
    40 absent from the CGM data.
    """
    return _raw("cgmacros_bio").sort_values("subject").reset_index(drop=True)


def gut_health() -> pd.DataFrame:
    """Viome pathway scores, 1=not optimal to 3=good.

    47 rows, not 45 — subjects 24 and 25 have gut panels but no CGM data, so an
    inner join on :func:`bio` is the right default rather than an outer one.
    """
    return _raw("cgmacros_gut_health").sort_values("subject").reset_index(drop=True)


def microbes() -> pd.DataFrame:
    """Binary presence/absence for 1,979 taxa, one row per subject (45 x 1,980)."""
    return _raw("cgmacros_microbes").sort_values("subject").reset_index(drop=True)


# ------------------------------------------------------------------ photos

def fetch_photos() -> Path:
    """Download the 213 MB photo bundle (once) and return the zip path."""
    return _fetch(PHOTO_ZIP)


def photo_members() -> frozenset[str]:
    """Every entry in the photo zip. Cached after the first call.

    Reads only the zip's central directory over HTTP ranges, so this does not
    pull 213 MB just to answer "does this photo exist".
    """
    global _MEMBERS
    if _MEMBERS is None:
        local = CACHE / PHOTO_ZIP
        if local.exists():
            with zipfile.ZipFile(local) as z:
                _MEMBERS = frozenset(z.namelist())
        else:
            with zipfile.ZipFile(_HttpFile(f"{BASE}/{PHOTO_ZIP}")) as z:
                _MEMBERS = frozenset(z.namelist())
    return _MEMBERS


_MEMBERS: frozenset[str] | None = None


def extract_photos(dest: Path | None = None) -> Path:
    """Extract the bundle once and return the directory holding it."""
    dest = Path(dest) if dest else CACHE / "photos"
    marker = dest / ".complete"
    if not marker.exists():
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(fetch_photos()) as z:
            z.extractall(dest)
        marker.touch()
    return dest


def photo_path(subject, image_path: str, *, root: Path | None = None) -> Path | None:
    """Resolve one meal photo to a file on disk, or None if it isn't in the zip."""
    member = zip_member(subject, image_path)
    if member not in photo_members():
        return None
    return extract_photos(root) / member


def photo_bytes(subject, image_path: str) -> bytes | None:
    """Read one photo straight out of the zip, without extracting everything."""
    member = zip_member(subject, image_path)
    if member not in photo_members():
        return None
    with zipfile.ZipFile(fetch_photos()) as z:
        return z.read(member)


class _HttpFile:
    """Minimal seekable read-only file over HTTP range requests."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, headers={"User-Agent": UA}, method="HEAD")
        self.size = int(urllib.request.urlopen(req).headers["Content-Length"])

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def seekable(self):
        return True

    def readable(self):
        return True

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        if n == 0:
            return b""
        end = min(self.pos + n, self.size) - 1
        headers = {"User-Agent": UA, "Range": f"bytes={self.pos}-{end}"}
        req = urllib.request.Request(self.url, headers=headers)
        data = urllib.request.urlopen(req).read()
        self.pos += len(data)
        return data


# ------------------------------------------------------------- Big Ideas Lab

def bigideas_cgm() -> pd.DataFrame:
    """Dexcom EGV readings at a 5-minute cadence, 16 participants."""
    df = _raw("bigideas_cgm")
    return df.sort_values(["participant_id", "timestamp"]).reset_index(drop=True)


def bigideas_wearable(*, min_beats: int = MIN_BEATS) -> pd.DataFrame:
    """Per-minute HR, RMSSD, EDA and skin temperature from an Empatica E4.

    ``rmssd_ms`` is computed per minute from however many beats were detected,
    and most minutes do not have enough: the median has 3 and 90,925 have none.
    ``rmssd_valid`` marks minutes with at least ``min_beats`` beats (30 leaves
    63,994 minutes, 30.6%). The column is flagged, not filtered -- decide for
    yourself what an HRV number built on 4 beats is worth.

    HR, EDA and skin temperature are each ~26% missing where the sensor dropped.
    """
    df = _raw("bigideas_wearable_1min").copy()
    df["rmssd_valid"] = df["n_beats"].fillna(0) >= min_beats
    return df.sort_values(["participant_id", "timestamp"]).reset_index(drop=True)


def bigideas_food() -> pd.DataFrame:
    """Food logs for 16 participants.

    ``amount`` is free text -- real values include "half", "rest of bar" and
    "1/2 cup". 293 of 1,422 rows have no numeric reading; use ``amount_num``,
    which is null for exactly those.

    ``schema_variant`` marks participant 003's 58 rows, whose source file had no
    header and only 11 columns. It was realigned upstream: the first nine fields
    map cleanly and the last two are kept as ``unmapped_a``/``unmapped_b``
    rather than guessed into ``sugar``/``protein``.
    """
    df = _raw("bigideas_food_log").copy()
    df["schema_variant"] = df["unmapped_a"].notna() | df["unmapped_b"].notna()
    return df.sort_values(["participant_id", "time_begin"]).reset_index(drop=True)


def bigideas_participants() -> pd.DataFrame:
    """Sex and HbA1c for the 16 Big Ideas participants (5.3-6.4%, pre-diabetic)."""
    return _raw("bigideas_participants").sort_values("participant_id").reset_index(drop=True)


# ---------------------------------------------------------------- NOTES
#
# Repairs applied here, and the evidence for each:
#
# 1. meal_type is normalised from ten published strings to four categories.
#    The file contains 'dinner' (418) and 'Dinner' (74), 'snack' (300),
#    'Snacks' (38), 'Snack' (4) and one 'snack 1'. One-hot encoding the raw
#    column yields ten columns and splits every category's signal. Casing is
#    also a per-subject habit -- subjects 1-4 capitalise throughout, 27 subjects
#    never do, 14 are mixed -- so the raw string partly encodes subject
#    identity. meal_type_raw is kept for anyone who wants to verify this.
#
# 2. Photo paths do not resolve as stored. The parquet holds
#    'photos/<basename>' while the zip is laid out 'CGMacros-<subject:03d>/
#    <basename>', so resolution needs the subject id. 3,194 of 3,197 referenced
#    photos resolve (99.91%). The 3 that do not belong to subjects 1 and 2 and
#    are flagged photo_ok=False rather than dropped -- and all three are
#    after-photos, so every one of the 1,644 meal photos resolves and the
#    modelling set loses nothing. 260 zip photos are referenced by no row at
#    all; they are harmless.
#
# 3. The timeseries carries 3,197 rows with an image_path but only 1,706 with
#    macros. The other 1,553 are post-meal photographs -- they follow a logged
#    meal by a median of 16 minutes, 87% within the hour. after_photos() keeps
#    them separate, because joining every image_path row to the meal table
#    produces 1,553 rows whose target is NaN.
#
# 4. Dexcom reads +31.95 mg/dL above Libre on average (median +32.20, sd 24.34,
#    r=0.830 across 629,605 minutes where both reported). Per-subject mean
#    offsets run from -24.3 to +71.7, and exactly one subject has Libre reading
#    higher. Libre covers 100% of rows and Dexcom 91.6%, so libre_gl is the
#    default. Pick one and say which; do not average them.
#
# 5. recordindex is 0.00% populated, sugar 0.01% and steps 0.82%. They are
#    dropped by default; pass keep_empty=True to retain them.
#
# 6. Subject ids are not contiguous -- 45 subjects with ids 1-49, missing 24,
#    25, 37 and 40. gut_health() has 47 rows because subjects 24 and 25 have
#    panels but no CGM data.
#
# 7. Big Ideas rmssd_ms is unusable on most rows: the median minute has 3
#    detected beats and 90,925 have zero. rmssd_valid flags n_beats >= 30.
#
# 8. Big Ideas participant 003's food log had a different schema; the two
#    columns that could not be matched to documented macros are unmapped_a and
#    unmapped_b, and schema_variant marks the 58 affected rows.
#
# 9. Big Ideas 'amount' is free text on 293 of 1,422 rows; amount_num carries
#    the numeric ones and is null for the rest.
#
# 10. `amount_consumed` mixes three incompatible scales in one column, and the
#     convention is per-subject. Across 1,642 non-null values: 15 subjects record
#     what look like percentages (50-100), 8 record small counts (0-9), and 16
#     have values above 100 running up to 900. The kit README calls it "how much
#     was actually eaten", which reads as a percentage and is true for only part
#     of the file. Because the scale is a per-subject habit the column also partly
#     encodes subject identity, the same trap as meal_type casing. It is left in
#     the frame -- it is real data and someone may be able to disentangle it per
#     subject -- but rung5_meal_risk.py excludes it from the feature set, and
#     doing so *improves* both heads (MAE 28.30 -> 28.17, AUC 0.8871 -> 0.8884).
#
# Licences differ between the two studies and it matters commercially:
# CGMacros is CC BY-NC-SA 4.0 (non-commercial, share-alike) -- Gutierrez-Osuna
# et al., PhysioNet 2025, doi:10.13026/3z8q-x658. Big Ideas Lab is ODC-BY 1.0
# (attribution only) -- Cho et al., PhysioNet, doi:10.13026/zthx-5212.
# Cite the original authors, not this repository or the bucket.
