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
