"""Temporary test file."""

import inspect

import pytest
import shap

from baybe.insights.shap import NON_SHAP_EXPLAINERS, _get_explainer_cls


def _has_required_init_parameters(cls: type[shap.Explainer]) -> bool:
    """Check if non-shap initializer has required standard parameters."""
    REQUIRED_PARAMETERS = ["self", "model", "data"]
    init_signature = inspect.signature(cls.__init__)
    parameters = list(init_signature.parameters.keys())
    return parameters[:3] == REQUIRED_PARAMETERS


@pytest.mark.parametrize("explainer_name", NON_SHAP_EXPLAINERS)
def test_non_shap_signature(explainer_name):
    """Non-SHAP explainers must have the required signature."""
    assert _has_required_init_parameters(_get_explainer_cls(explainer_name))
