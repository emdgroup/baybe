"""Tests for the targets module."""

import pytest

from baybe.targets import NumericalTarget, Objective
from baybe.utils import StrictValidationError
from pydantic import ValidationError

# Validation errors to be caught
validation_errors = (ValidationError, StrictValidationError)

# Two example targets used in the tests
two_targets = [
    NumericalTarget(
        name="Target_1",
        mode="MAX",
        bounds=(0, 100),
    ),
    NumericalTarget(
        name="Target_2",
        mode="MIN",
        bounds=(0, 100),
    ),
]


def test_invalid_target_creation():
    """Invalid target creation raises expected error."""

    # Scenario: missing bounds for match mode
    with pytest.raises(validation_errors):
        NumericalTarget(
            name="missing_bounds",
            mode="MATCH",
        )

    # Scenario: incompatible transformation for match mode
    with pytest.raises(validation_errors):
        NumericalTarget(
            name="incompatible_transform",
            mode="MATCH",
            bounds=(0, 100),
            bounds_transform_func="LINEAR",
        )

    # Scenario: invalid transformation
    with pytest.raises(validation_errors):
        NumericalTarget(
            name="invalid_transform",
            mode="MATCH",
            bounds=(0, 100),
            bounds_transform_func="SOME_STUFF",
        )


def test_invalid_objective_creation():
    """Invalid objective creation raises expected error."""

    # Scenario: empty target list
    with pytest.raises(validation_errors):
        Objective(
            mode="SINGLE",
            targets=[],
        )

    # Scenario: wrong target type
    with pytest.raises(validation_errors):
        Objective(
            mode="SINGLE",
            targets={"A": 1, "B": 2},
        )

    # Scenario: missing bounds for desirability
    with pytest.raises(validation_errors):
        Objective(
            mode="DESIRABILITY",
            targets=[
                NumericalTarget(
                    name="Target_1",
                    mode="MAX",
                    bounds=(0, 100),
                ),
                NumericalTarget(
                    name="Target_2",
                    mode="MIN",
                ),
            ],
        )

    # Scenario: invalid combination function
    with pytest.raises(validation_errors):
        Objective(
            mode="DESIRABILITY",
            combine_func="FALSE_STUFF",
            targets=two_targets,
        )

    # Scenario: invalid weights
    with pytest.raises(validation_errors):
        Objective(
            mode="DESIRABILITY",
            weights=[1, 2, 3],
            targets=two_targets,
        )

    # Scenario: invalid weights
    with pytest.raises(validation_errors):
        Objective(
            mode="DESIRABILITY",
            weights=[1, "ABC"],
            targets=two_targets,
        )

    # Scenario: invalid weights
    with pytest.raises(validation_errors):
        Objective(
            mode="DESIRABILITY",
            weights="ABC",
            targets=two_targets,
        )
