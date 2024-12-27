"""Baybe insights (optional)."""

from baybe._optional.info import INSIGHTS_INSTALLED

if INSIGHTS_INSTALLED:
    from baybe.insights.shap import SHAPInsight

__all__ = [
    "SHAPInsight",
]
