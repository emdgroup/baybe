"""Validation tests for constraints."""

import pytest
from pytest import param

from baybe.constraints.conditions import ThresholdCondition
from baybe.constraints.continuous import ContinuousCardinalityConstraint
from baybe.constraints.discrete import DiscreteSumConstraint


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


def test_discrete_sum_constraint_coefficients_length_mismatch():
    """Mismatched coefficients length raises a ValueError."""
    with pytest.raises(ValueError, match="'coefficients' list must have one"):
        DiscreteSumConstraint(
            parameters=["A", "B", "C"],
            condition=ThresholdCondition(threshold=1.0, operator="<="),
            coefficients=(1.0, 2.0),  # only 2 entries for 3 parameters
        )
