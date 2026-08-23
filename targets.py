"""targets.py — turn logged meals into a supervised glucose-response target.

    import targets
    w = targets.meal_windows()          # one row per meal, with iauc
    d = targets.modelling_set()         # + photo, macros, clean window only

The label is the **incremental AUC** of the glucose curve over the two hours
after a meal: the area between the curve and its own starting value, clipped at
zero. That is the standard construction in the glycemic literature and it is
less sensitive to a single noisy reading than the peak is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import cgm

WINDOW_MIN = 120
MACROS = ("carbs", "protein", "fat", "fiber", "calories")

# A window is only as good as its coverage. Below this fraction of the minutes
# present, the trapezoid is integrating over guesswork.
MIN_COVERAGE = 0.8


def meal_windows(
    *,
    window_min: int = WINDOW_MIN,
    sensor: str = cgm.DEFAULT_SENSOR,
    min_coverage: float = MIN_COVERAGE,
) -> pd.DataFrame:
    """One row per logged meal with its post-meal glucose response.

    Columns added to the meal row:

    ``baseline``      glucose at the meal timestamp (mg/dL)
    ``peak``          maximum over the window
    ``peak_delta``    peak minus baseline
    ``iauc``          incremental AUC, mg/dL·h — **the target**
    ``n_obs``         glucose readings present in the window
    ``coverage``      fraction of the window's minutes that reported
    ``clean_window``  no other logged meal starts inside the window
    ``next_gap_min``  minutes to the next logged meal, NaN if none

    ``peak_delta`` and ``iauc`` correlate 0.953, so they are near
    interchangeable; ``iauc`` is the primary and ``peak_delta`` is kept as a
    secondary that is easier to explain to a clinician.
    """
    ts = cgm.timeseries(sensor=sensor)
    meals = cgm.meals(with_photo=True)

    meals = meals.sort_values(["subject", "timestamp"]).copy()
    nxt = meals.groupby("subject")["timestamp"].shift(-1)
    meals["next_gap_min"] = (nxt - meals["timestamp"]).dt.total_seconds() / 60
    meals["clean_window"] = meals["next_gap_min"].isna() | (
        meals["next_gap_min"] >= window_min
    )

    span = pd.Timedelta(minutes=window_min)
    expected = window_min + 1
    rows = []
    for subject, group in ts.groupby("subject", sort=False):
        series = group.set_index("timestamp")["glucose"].sort_index()
        for t in meals.loc[meals.subject == subject, "timestamp"]:
            window = series.loc[t : t + span].dropna()
            if window.empty:
                rows.append((subject, t, np.nan, np.nan, np.nan, np.nan, 0, 0.0))
                continue
            baseline = float(window.iloc[0])
            delta = (window - baseline).clip(lower=0)
            minutes = (window.index - window.index[0]).total_seconds() / 60
            iauc = float(np.trapezoid(delta.to_numpy(), minutes) / 60)
            rows.append((
                subject, t, baseline, float(window.max()),
                float(window.max()) - baseline, iauc,
                len(window), len(window) / expected,
            ))

    resp = pd.DataFrame(rows, columns=[
        "subject", "timestamp", "baseline", "peak", "peak_delta",
        "iauc", "n_obs", "coverage",
    ])
    out = meals.merge(resp, on=["subject", "timestamp"], how="left")
    out["well_covered"] = out["coverage"] >= min_coverage
    return out.reset_index(drop=True)


def modelling_set(
    *,
    window_min: int = WINDOW_MIN,
    sensor: str = cgm.DEFAULT_SENSOR,
    require_photo: bool = True,
    require_clean: bool = True,
) -> pd.DataFrame:
    """The rows every rung actually trains on, with the filtering made explicit.

    Defaults to meals that have a photo, complete macros, a well-covered window
    and no second meal inside it. Each filter is separately switchable so a
    sensitivity check is one argument away rather than a rewrite.

    Prints the funnel, because the honest n is smaller than the headline 1,706
    and it should be visible every single run rather than quoted from a README.
    """
    df = meal_windows(window_min=window_min, sensor=sensor)
    steps = [("logged meals", len(df))]

    df = df[df["well_covered"]]
    steps.append((f"window >= {int(100 * MIN_COVERAGE)}% covered", len(df)))

    df = df[df[list(MACROS)].notna().all(axis=1)]
    steps.append(("complete macros", len(df)))

    if require_photo:
        df = df[df["photo_ok"]]
        steps.append(("photo resolves", len(df)))
    if require_clean:
        df = df[df["clean_window"]]
        steps.append((f"no second meal within {window_min}min", len(df)))

    print(f"modelling set ({sensor}, {window_min}min window)")
    prev = None
    for label, n in steps:
        drop = "" if prev is None else f"  (-{prev - n})"
        print(f"  {label:<38} {n:>6,}{drop}")
        prev = n
    print(f"  {'subjects':<38} {df.subject.nunique():>6}")
    return df.reset_index(drop=True)


def curve(subject, timestamp, *, window_min: int = WINDOW_MIN,
          sensor: str = cgm.DEFAULT_SENSOR) -> pd.DataFrame:
    """The observed glucose curve for one meal, as minute/delta pairs.

    Used to render a real response next to a predicted one.
    """
    ts = cgm.timeseries(sensor=sensor)
    series = ts[ts.subject == subject].set_index("timestamp")["glucose"].sort_index()
    window = series.loc[timestamp : timestamp + pd.Timedelta(minutes=window_min)].dropna()
    if window.empty:
        return pd.DataFrame(columns=["minute", "glucose", "delta"])
    minutes = (window.index - window.index[0]).total_seconds() / 60
    return pd.DataFrame({
        "minute": minutes,
        "glucose": window.to_numpy(),
        "delta": window.to_numpy() - float(window.iloc[0]),
    })
