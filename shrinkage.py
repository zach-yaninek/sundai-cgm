"""The personalisation maths, with no dependencies worth carrying.

Split out of `personalize.py` because that module imports pandas, xgboost and
the training code to *measure* the learning curve — none of which the serving
path needs to *apply* the result. Without this split the API container fails at
import, which is exactly how it was found.

    from shrinkage import shrinkage, offset_from_residuals
"""
from __future__ import annotations

from math import isfinite

# One logged meal moves the correction 1/6 of the way, five moves it half, and
# fifteen three quarters. Chosen because the measured learning curve flattens
# around k=5-10, so a constant reaching half weight there tracks the evidence.
LAMBDA = 5.0


def shrinkage(k: int, lam: float = LAMBDA) -> float:
    """Weight given to a person's own history after ``k`` logged meals."""
    return 0.0 if k <= 0 else k / (k + lam)


def offset_from_residuals(residuals, lam: float = LAMBDA) -> float:
    """The correction to add to a population prediction, already shrunk.

    ``residuals`` are ``observed - predicted`` for meals the person has logged.
    Returns 0.0 for an empty history — an app that "personalises" before it has
    evidence is inventing the thing it claims to have learned.
    """
    usable = [float(r) for r in residuals
              if r is not None and isfinite(float(r))]
    if not usable:
        return 0.0
    mean = sum(usable) / len(usable)
    return mean * shrinkage(len(usable), lam)


# Below this many logged meals, only an intercept is learned. A two-parameter
# fit on three points is mostly fitting which three meals happened to be logged,
# and personalize_compare.py measured exactly that: at k=3 the slope scores 28.00
# against the intercept-only 25.77, at k=4 26.14 against 25.38, at k=5 they tie,
# and from k=6 the slope leads (24.43 against 24.74) and never gives it back.
# Six is where the sweep crossed over, not a round number that looked safe.
SLOPE_MIN_POINTS = 6


class Calibration:
    """What has been learned about one person, as a correction to any prediction.

    Two parameters against the population model's output:

        corrected = (1 - w) * predicted + w * (intercept + slope * predicted)

    which rearranges to a per-prediction offset of
    ``w * (intercept + (slope - 1) * predicted)``. An intercept-only calibration
    is exactly ``slope = 1``, where that collapses to ``w * intercept`` — the
    scalar correction this module started with — so both regimes are one formula
    and there is no second code path to keep in step.

    The slope is what an intercept cannot express: not "the model runs 20 low for
    you" but "the model understates your *large* responses specifically".
    """

    __slots__ = ("intercept", "slope", "weight", "k")

    def __init__(self, intercept: float = 0.0, slope: float = 1.0,
                 weight: float = 0.0, k: int = 0):
        self.intercept = intercept
        self.slope = slope
        self.weight = weight
        self.k = k

    @property
    def learned_slope(self) -> bool:
        """Whether this fit has a slope, or is the intercept-only regime."""
        return self.slope != 1.0

    def offset_for(self, predicted: float) -> float:
        """mg/dL*h to add to this particular prediction."""
        return self.weight * (self.intercept + (self.slope - 1.0) * predicted)

    def apply(self, predicted: float) -> float:
        return predicted + self.offset_for(predicted)

    def __repr__(self) -> str:
        return (f"Calibration(intercept={self.intercept:.2f}, "
                f"slope={self.slope:.3f}, weight={self.weight:.3f}, k={self.k})")


def fit_calibration(predicted, observed, lam: float = LAMBDA,
                    min_slope_points: int = SLOPE_MIN_POINTS) -> Calibration:
    """Learn a person's correction from what the model said and what happened.

    Ordinary least squares, written out rather than imported, so this module
    stays stdlib-only and the serving container does not gain a dependency to
    fit two parameters to fifteen points.

    Falls back to intercept-only when there are too few meals to support a slope,
    and also when the predictions carry no spread — a slope through a vertical
    stack of points is a division by nearly zero wearing a number's clothing.
    """
    pairs = [(float(p), float(o)) for p, o in zip(predicted, observed)
             if p is not None and o is not None
             and isfinite(float(p)) and isfinite(float(o))]
    n = len(pairs)
    if n == 0:
        return Calibration()

    weight = shrinkage(n, lam)
    mean_residual = sum(o - p for p, o in pairs) / n

    if n < min_slope_points:
        return Calibration(intercept=mean_residual, slope=1.0, weight=weight, k=n)

    mean_p = sum(p for p, _ in pairs) / n
    variance = sum((p - mean_p) ** 2 for p, _ in pairs)
    if variance < 1e-9:
        return Calibration(intercept=mean_residual, slope=1.0, weight=weight, k=n)

    mean_o = sum(o for _, o in pairs) / n
    covariance = sum((p - mean_p) * (o - mean_o) for p, o in pairs)
    slope = covariance / variance
    intercept = mean_o - slope * mean_p
    return Calibration(intercept=intercept, slope=slope, weight=weight, k=n)
