"""Validation tests for constraints."""

import pytest
from pytest import param

from baybe.constraints.conditions import ThresholdCondition
from baybe.constraints.continuous import (
    ContinuousCardinalityConstraint,
    ContinuousLinearConstraint,
)
from baybe.constraints.discrete import (
    DiscreteRepetitionConstraint,
    DiscreteSumConstraint,
)


@pytest.mark.parametrize(
    ("cardinalities", "error", "match"),
    [
        param(("0", 0), TypeError, "must be <class 'int'>", id="type_min"),
        param((0, "1"), TypeError, "must be <class 'int'>", id="type_max"),
        param((-1, 0), ValueError, "'min_cardinality' must be >= 0", id="loo_small"),
        param((1, 0), ValueError, "larger than the upper bound", id="wrong_order"),
        param((0, 3), ValueError, "exceed the number of parameters", id="too_large"),
        param((0, 2), ValueError, r"No constraint .* required", id="inactive"),
    ],
)
def test_invalid_cardinalities(cardinalities, error, match):
    """Providing an invalid parameter name raises an exception."""
    with pytest.raises(error, match=match):
        ContinuousCardinalityConstraint(["x", "y"], *cardinalities)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        param(
            {"n_max_repetitions": 2.0},
            TypeError,
            "must be <class 'int'>",
            id="maximum-type",
        ),
        param(
            {"n_max_repetitions": 0},
            ValueError,
            "must be >= 1",
            id="maximum-too-small",
        ),
        param(
            {"n_max_repetitions": 4},
            ValueError,
            "must be less than the number of parameters",
            id="maximum-too-large",
        ),
        param(
            {"n_max_repetitions": 3},
            ValueError,
            "meaningful constraint",
            id="maximum-only-no-op",
        ),
    ],
)
def test_invalid_max_repetitions(kwargs, error, match):
    """Invalid maximum repetition counts raise an exception."""
    with pytest.raises(error, match=match):
        DiscreteRepetitionConstraint(parameters=["A", "B", "C"], **kwargs)


@pytest.mark.parametrize(
    ("coefficients", "match"),
    [
        param((1.0, 2.0), "'coefficients' list must have one", id="length-mismatch"),
        param((1.0, 0.0, 1.0), "'coefficients' must be non-zero", id="zero-coeff"),
    ],
)
def test_invalid_coefficients(coefficients, match):
    """Invalid coefficients raise a ValueError."""
    with pytest.raises(ValueError, match=match):
        DiscreteSumConstraint(
            parameters=["A", "B", "C"],
            condition=ThresholdCondition(threshold=1.0, operator="<="),
            coefficients=coefficients,
        )
    with pytest.raises(ValueError, match=match):
        ContinuousLinearConstraint(
            parameters=["A", "B", "C"],
            operator="<=",
            coefficients=coefficients,
        )
