"""Tests for fidelity parameters."""

import pandas as pd
import pytest
from pytest import param

from baybe.parameters.categorical import TaskParameter
from baybe.parameters.fidelity import (
    CategoricalFidelityParameter,
    NumericalDiscreteFidelityParameter,
)
from baybe.searchspace.core import SearchSpace


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
    ("parameter", "series", "expected"),
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
def test_fidelity_parameter_transform(parameter, series, expected):
    """Transform must correctly map fidelity values to computational representation."""
    result = parameter.transform(series)
    assert list(result["fidelity"]) == expected


@pytest.mark.parametrize(
    ("parameters", "match"),
    [
        param(
            [
                CategoricalFidelityParameter(
                    "f1", values=["lo", "hi"], costs=[1, 10], zeta=[0.5, 0.0]
                ),
                CategoricalFidelityParameter(
                    "f2", values=["a", "b"], costs=[1, 5], zeta=[0.3, 0.0]
                ),
            ],
            "at most one fidelity",
            id="two_categorical_fidelity",
        ),
        param(
            [
                CategoricalFidelityParameter(
                    "f1", values=["lo", "hi"], costs=[1, 10], zeta=[0.5, 0.0]
                ),
                NumericalDiscreteFidelityParameter(
                    "f2", values=[0.5, 1.0], costs=[1, 10]
                ),
            ],
            "at most one fidelity",
            id="mixed_fidelity_types",
        ),
        param(
            [
                NumericalDiscreteFidelityParameter(
                    "f1", values=[0.5, 1.0], costs=[1, 10]
                ),
                NumericalDiscreteFidelityParameter(
                    "f2", values=[0.2, 1.0], costs=[1, 5]
                ),
            ],
            "at most one fidelity",
            id="two_numerical_fidelity",
        ),
        param(
            [
                TaskParameter("task", values=["a", "b"]),
                CategoricalFidelityParameter(
                    "f", values=["lo", "hi"], costs=[1, 10], zeta=[0.5, 0.0]
                ),
            ],
            "Combining task.*fidelity",
            id="task_plus_fidelity",
        ),
    ],
)
def test_invalid_fidelity_parameter_combinations(parameters, match):
    """Search spaces with invalid fidelity parameter combinations are rejected."""
    with pytest.raises(NotImplementedError, match=match):
        SearchSpace.from_product(parameters)
