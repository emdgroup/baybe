"""Baybe diagnostics (optional)."""

from baybe._optional.info import DIAGNOSTICS_INSTALLED

if DIAGNOSTICS_INSTALLED:
    from baybe.diagnostics.shap import explainer, explanation, plot_shap_scatter

__all__ = [
    "explainer",
    "explanation",
    "plot_shap_scatter",
]
