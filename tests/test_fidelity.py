"""Tests for fidelity parameters."""

import pandas as pd
import pytest
from pytest import param

from baybe.parameters.fidelity import (
    CategoricalFidelityParameter,
    NumericalDiscreteFidelityParameter,
)


def test_categorical_fidelity_parameter_construction():
    """Equivalent zeta formats and value orderings produce equal objects."""
    p1 = CategoricalFidelityParameter("p", values=["h", "l"], costs=[1, 2], zeta=5)
    p2 = CategoricalFidelityParameter("p", values=["l", "h"], costs=[2, 1], zeta=[5, 0])
    assert p1 == p2


def test_numerical_discrete_fidelity_parameter_construction():
    """Fidelity values and costs are sorted according to numerical fidelity values."""
    p1 = NumericalDiscreteFidelityParameter("p", values=[0, 0.5, 1], costs=[1, 2, 3])
    p2 = NumericalDiscreteFidelityParameter("p", values=[0.5, 1, 0], costs=[2, 3, 1])
    assert p1 == p2


@pytest.mark.parametrize(
    ("param", "series", "expected"),
    [
        param(
            CategoricalFidelityParameter(
                "fidelity", values=["low", "high"], costs=[1, 2], zeta=[1, 0]
            ),
            pd.Series(["low", "high", "low"], name="fidelity"),
            [1.0, 0.0, 1.0],
            id="categorical",
        ),
        param(
            CategoricalFidelityParameter(
                "fidelity", values=["low", "high"], costs=[1, 2], zeta=5
            ),
            pd.Series(["low", "high", "low"], name="fidelity"),
            [1.0, 0.0, 1.0],
            id="categorical_scalar_zeta",
        ),
        param(
            NumericalDiscreteFidelityParameter(
                "fidelity", values=[0, 0.5, 1], costs=[1, 2, 3]
            ),
            pd.Series([0.5, 1.0, 0.0], name="fidelity"),
            [0.5, 1.0, 0.0],
            id="numerical_discrete",
        ),
    ],
)
def test_fidelity_parameter_transform(param, series, expected):
    """Transform must correctly map fidelity values to computational representation."""
    result = param.transform(series)
    assert list(result["fidelity"]) == expected
