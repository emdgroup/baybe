"""Temporary test file."""

import inspect

import numpy as np
import pandas as pd
import pytest

from baybe._optional.info import INSIGHTS_INSTALLED

if not INSIGHTS_INSTALLED:
    pytest.skip("Optional insights package not installed.", allow_module_level=True)

import shap
from shap.explainers import KernelExplainer

from baybe.insights.shap import NON_SHAP_EXPLAINERS, SHAPInsight, _get_explainer_cls


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


def test_column_permutation():
    """Explaining data with permuted columns gives permuted explanations."""
    N = 10

    # Create insights object and test data
    background_data = pd.DataFrame(np.random.random((N, 3)), columns=["x", "y", "z"])
    explainer = KernelExplainer(lambda x: x, background_data)
    insights = SHAPInsight(explainer, background_data)
    df = pd.DataFrame(np.random.random((N, 3)), columns=["x", "y", "z"])

    # Regular column order
    ex1 = insights.explain(df)

    # Permuted column order
    ex2 = insights.explain(df[["z", "x", "y"]])[:, [1, 2, 0]]

    assert np.array_equal(ex1.values, ex2.values)
