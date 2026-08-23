"""Glucose-response modelling on CGMacros.

The installable surface is deliberately small. `value_of_information` answers
"would drawing labs tell us anything about this person" from a measured grid, and
needs no model weights and no xgboost:

    from sundai_cgm import value_of_information
    value_of_information({"a1c_pdl_lab": 6.2}, pre_meal_glucose=104)

Full per-meal scoring lives in the repository's `risk.py` and needs the model
artifacts; install the `[scoring]` extra and work from a checkout for that.
"""
from .voi import (
    TIERS,
    best_tier,
    grid,
    reliability,
    tier_for,
    value_of_information,
)

__all__ = ["value_of_information", "reliability", "tier_for", "best_tier",
           "grid", "TIERS"]
__version__ = "1.1.0"
