"""personalize_compare.py — is one number per person really the best we can do?

The app corrects a population prediction by a single shrunk scalar: the mean gap
between what happened to you and what was predicted. That is the cheapest
possible way to "learn a patient", and the obvious question is whether something
with more parameters beats it.

This measures that rather than arguing it, under the same discipline as
`personalize.py`: leave one subject out, train the population model on everyone
else, reveal `k` of the held-out person's meals to fit whatever the strategy
learns, and score on the meals that remain. Calibration meals never appear in
the evaluation set, or the answer is memorisation.

Two things make this a fair fight rather than a demonstration:

**Every strategy sees the identical split.** Same subject, same permutation,
same k, same evaluation rows. Differences are the strategy, not the draw.

**Repeats, because one permutation per subject is noisy.** Which k meals happen
to be revealed matters a lot at k=1 and still matters at k=5, so each subject is
run over several shuffles and the results averaged.

Intervals are bootstrapped over *subjects*, matching evaluate.py, because the
effective n here is people and not meals.

    python personalize_compare.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

import rung5_meal_risk as r5
from shrinkage import LAMBDA, offset_from_residuals, shrinkage

ARTIFACTS = Path(__file__).parent / "artifacts"

CURVE_K = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15)
MIN_MEALS = 25
HOLDOUT = 5
REPEATS = 5
N_BOOT = 2000

# A per-user booster sees at most 15 rows. Handing it the population config
# (400 trees, depth 4) would be testing a straw man, so it gets a configuration
# someone would actually choose for tiny data.
USER_REG_PARAMS = dict(
    n_estimators=100, max_depth=2, learning_rate=0.1,
    subsample=1.0, colsample_bytree=1.0, reg_lambda=5.0, min_child_weight=1,
    objective="reg:squarederror", random_state=0, n_jobs=1,
)


# ------------------------------------------------------------------ strategies
#
# Each takes the calibration slice and returns adjusted predictions for the
# evaluation slice. All of them shrink toward the population model, because the
# thing being tested is what to learn, not whether to shrink.

def s_none(ctx) -> np.ndarray:
    return ctx["pred_eval"]


def s_offset(ctx) -> np.ndarray:
    """What the app does today: one shrunk scalar."""
    return ctx["pred_eval"] + offset_from_residuals(ctx["resid_cal"])


def s_offset_by_meal_type(ctx) -> np.ndarray:
    """A separate shrunk offset per meal type, backing off to the global one.

    The premise is that someone might tolerate breakfast badly and dinner fine.
    Each meal type's offset is shrunk by ITS OWN count, so a type seen once
    barely moves, and types never seen fall back to the global offset.
    """
    global_offset = offset_from_residuals(ctx["resid_cal"])
    out = ctx["pred_eval"] + global_offset
    if len(ctx["resid_cal"]) == 0:
        return out
    for meal_type in np.unique(ctx["mt_cal"]):
        seen = ctx["resid_cal"][ctx["mt_cal"] == meal_type]
        target = ctx["mt_eval"] == meal_type
        if target.any():
            out[target] = ctx["pred_eval"][target] + offset_from_residuals(seen)
    return out


def _slope_intercept(ctx, min_points: int) -> np.ndarray:
    """Learn a scale as well as a shift: observed ~ a + b * predicted.

    A scalar offset can only say "the model runs 20 low for you". This can say
    "the model understates your big meals specifically", which is a different
    and plausible shape of error. Shrunk toward the identity line (a=0, b=1).

    ``min_points`` is the whole ballgame. A two-parameter fit on three points is
    mostly fitting the draw, and the first pass measured exactly that: enabled at
    k=3 it scored 28.00 against the scalar offset's 25.77, wiping out the gain it
    delivers later. Below the gate this falls back to the scalar.
    """
    cal, resid = ctx["pred_cal"], ctx["resid_cal"]
    if len(cal) < min_points or np.ptp(cal) < 1e-6:
        return s_offset(ctx)
    b, a = np.polyfit(cal, cal + resid, 1)
    weight = shrinkage(len(cal))
    corrected = a + b * ctx["pred_eval"]
    return (1 - weight) * ctx["pred_eval"] + weight * corrected


# Where the two-parameter fit starts beating the one-parameter one, measured
# rather than chosen: at k=3 the slope scores 28.00 against the scalar's 25.77,
# at k=4 26.14 against 25.38, at k=5 they tie, and from k=6 the slope leads
# (24.43 against 24.74) and never gives the lead back.
SLOPE_MIN_POINTS = 6


def s_slope_ungated(ctx) -> np.ndarray:
    return _slope_intercept(ctx, 3)


def s_slope_gated(ctx) -> np.ndarray:
    return _slope_intercept(ctx, SLOPE_MIN_POINTS)


def s_user_booster(ctx) -> np.ndarray:
    """Retrain from scratch on this person's meals, blended by the same weight.

    This is "retrain per user" taken seriously: a real gradient-boosted model
    fitted to their own logged meals, then blended with the population model by
    the same k/(k+5) so it is not being penalised for a lack of shrinkage the
    others get.
    """
    if len(ctx["y_cal"]) < 2:
        return s_offset(ctx)
    model = xgb.XGBRegressor(**USER_REG_PARAMS)
    model.fit(ctx["X_cal"], ctx["y_cal"], verbose=False)
    weight = shrinkage(len(ctx["y_cal"]))
    return (1 - weight) * ctx["pred_eval"] + weight * model.predict(ctx["X_eval"])


def s_user_booster_residual(ctx) -> np.ndarray:
    """Learn the person's RESIDUAL from features, not their response from scratch.

    Strictly more informed than s_user_booster: the population model keeps doing
    the heavy lifting and the per-user model only has to explain what it got
    wrong, which is a much smaller function to fit on 15 rows.
    """
    if len(ctx["y_cal"]) < 2:
        return s_offset(ctx)
    model = xgb.XGBRegressor(**USER_REG_PARAMS)
    model.fit(ctx["X_cal"], ctx["resid_cal"], verbose=False)
    weight = shrinkage(len(ctx["resid_cal"]))
    return ctx["pred_eval"] + weight * model.predict(ctx["X_eval"])


STRATEGIES = {
    "none": s_none,
    "offset (current)": s_offset,
    "offset by meal type": s_offset_by_meal_type,
    "slope, ungated": s_slope_ungated,
    "slope, gated k>=6": s_slope_gated,
    "per-user booster": s_user_booster,
    "per-user residual booster": s_user_booster_residual,
}


# ------------------------------------------------------------------- the sweep

def run(*, target: str = "iauc", with_glucose: bool = True, seed: int = 0):
    df = r5.build_frame()
    X, _ = r5.build_features(df, with_glucose=with_glucose)
    y = df[target].to_numpy(dtype=float)
    groups = df["subject"].to_numpy()
    meal_types = df["meal_type"].to_numpy()

    subjects = [s for s in np.unique(groups) if (groups == s).sum() >= MIN_MEALS]
    print(f"{len(subjects)} subjects with >= {MIN_MEALS} meals, "
          f"{REPEATS} shuffles each, target={target}\n")

    rows = []
    started = time.time()
    for i, subject in enumerate(subjects, 1):
        train = groups != subject
        test = np.flatnonzero(groups == subject)

        model = xgb.XGBRegressor(**r5.REG_PARAMS)
        model.fit(X[train], y[train], verbose=False)
        pred = model.predict(X.iloc[test])
        resid = y[test] - pred

        X_subject = X.iloc[test].to_numpy()
        y_subject = y[test]
        mt_subject = meal_types[test]

        rng = np.random.default_rng(seed + int(subject))
        for repeat in range(REPEATS):
            order = rng.permutation(len(test))
            for k in CURVE_K:
                if k > len(test) - HOLDOUT:
                    continue
                cal, ev = order[:k], order[k:]
                ctx = {
                    "X_cal": X_subject[cal], "X_eval": X_subject[ev],
                    "y_cal": y_subject[cal], "y_eval": y_subject[ev],
                    "pred_cal": pred[cal], "pred_eval": pred[ev],
                    "resid_cal": resid[cal],
                    "mt_cal": mt_subject[cal], "mt_eval": mt_subject[ev],
                }
                for name, fn in STRATEGIES.items():
                    mae = float(np.abs(fn(ctx) - ctx["y_eval"]).mean())
                    rows.append({"subject": int(subject), "repeat": repeat,
                                 "k": k, "strategy": name, "mae": mae})
        print(f"  [{i:>2}/{len(subjects)}] subject {int(subject):>3} "
              f"({time.time() - started:.0f}s)", flush=True)

    return pd.DataFrame(rows)


def boot_ci(per_subject: pd.DataFrame, seed: int = 0):
    """Bootstrap over subjects. The effective n is people, not meals."""
    rng = np.random.default_rng(seed)
    values = per_subject.to_numpy(dtype=float)
    draws = rng.choice(len(values), size=(N_BOOT, len(values)), replace=True)
    stats = values[draws].mean(axis=1)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    raw = run()

    # Average the shuffles within a subject first, so every subject weighs the
    # same regardless of how many meals they logged.
    by_subject = (raw.groupby(["strategy", "k", "subject"])["mae"]
                  .mean().reset_index())
    table = (by_subject.groupby(["strategy", "k"])["mae"]
             .mean().reset_index()
             .pivot(index="k", columns="strategy", values="mae"))
    order = list(STRATEGIES)
    table = table[order]

    print("\n" + "=" * 78)
    print("MAE (mg/dL*h) by strategy and meals logged — lower is better")
    print("=" * 78)
    print(table.round(2).to_string())

    baseline = table.loc[0, "none"]
    print("\nimprovement vs no personalisation (%)")
    print(((baseline - table) / baseline * 100).round(1).to_string())

    print("\n" + "=" * 78)
    print(f"at k=15, bootstrapped over subjects ({N_BOOT} draws)")
    print("=" * 78)
    summary = {}
    for name in order:
        sel = by_subject[(by_subject.strategy == name) & (by_subject.k == 15)]
        if sel.empty:
            continue
        mean = float(sel["mae"].mean())
        lo, hi = boot_ci(sel["mae"])
        summary[name] = {"mae": round(mean, 3), "ci": [round(lo, 3), round(hi, 3)]}
        print(f"  {name:<28} {mean:6.2f}   [{lo:.2f}, {hi:.2f}]")

    # Paired comparison: same subject, same shuffles, so the difference per
    # subject is meaningful in a way the two marginal intervals are not.
    print("\npaired against the current scalar offset, at k=15")
    current = by_subject[(by_subject.strategy == "offset (current)")
                         & (by_subject.k == 15)].set_index("subject")["mae"]
    for name in order:
        if name in ("none", "offset (current)"):
            continue
        other = by_subject[(by_subject.strategy == name)
                           & (by_subject.k == 15)].set_index("subject")["mae"]
        diff = (other - current).dropna()
        if diff.empty:
            continue
        lo, hi = boot_ci(diff)
        verdict = "better" if hi < 0 else "worse" if lo > 0 else "no difference"
        print(f"  {name:<28} {diff.mean():+6.2f}   [{lo:+.2f}, {hi:+.2f}]   {verdict}")
        summary.setdefault(name, {})["vs_current"] = {
            "delta": round(float(diff.mean()), 3),
            "ci": [round(lo, 3), round(hi, 3)],
            "verdict": verdict,
        }

    ARTIFACTS.mkdir(exist_ok=True)
    payload = {
        "target": "iauc",
        "units": "mg/dL*h",
        "lambda": LAMBDA,
        "method": ("leave-one-subject-out; k meals revealed to fit each strategy, "
                   f"scored on that subject's remaining meals; {REPEATS} shuffles "
                   "per subject, identical splits across strategies"),
        "n_subjects": int(by_subject.subject.nunique()),
        "min_meals_per_subject": MIN_MEALS,
        "repeats": REPEATS,
        "curve": {name: {int(k): round(float(table.loc[k, name]), 3)
                         for k in table.index}
                  for name in order},
        "at_k15": summary,
    }
    (ARTIFACTS / "personalize_compare.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {ARTIFACTS / 'personalize_compare.json'}")


if __name__ == "__main__":
    main()
