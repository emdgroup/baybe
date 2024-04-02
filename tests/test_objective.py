"""Tests for the objective module."""

import numpy as np
import pytest
from cattrs import IterableValidationError

from baybe.objectives.desirability import DesirabilityObjective, scalarize
from baybe.objectives.enum import Scalarization
from baybe.objectives.single import SingleTargetObjective
from baybe.targets import NumericalTarget


class TestInvalidObjectiveCreation:
    """Invalid objective creation raises expected error."""

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

    def test_empty_target_list(self):
        with pytest.raises(ValueError):
            DesirabilityObjective(targets=[])

    def test_wrong_target_type(self):
        with pytest.raises(TypeError):
            SingleTargetObjective(target={"A": 1, "B": 2})

    def test_missing_bounds_for_desirability(self):
        with pytest.raises(ValueError):
            DesirabilityObjective(
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

    def test_invalid_combination_function(self):
        with pytest.raises(ValueError):
            DesirabilityObjective(
                targets=self.two_targets,
                scalarization="FALSE_STUFF",
            )

    def test_wrong_number_of_weights(self):
        with pytest.raises(ValueError):
            DesirabilityObjective(
                targets=self.two_targets,
                weights=[1, 2, 3],
            )

    def test_non_numeric_weights(self):
        with pytest.raises(IterableValidationError):
            DesirabilityObjective(
                targets=self.two_targets,
                weights=[1, "ABC"],
            )

    def test_wrong_weights_type(self):
        with pytest.raises(IterableValidationError):
            DesirabilityObjective(
                targets=self.two_targets,
                weights="ABC",
            )


@pytest.mark.parametrize(
    ("values", "scalarization", "weights", "expected"),
    [
        ([[1, 2]], Scalarization.MEAN, [1, 1], [1.5]),
        ([[1, 2]], Scalarization.MEAN, [1, 2], [5 / 3]),
        ([[1, 2]], Scalarization.GEOM_MEAN, [1, 1], [np.sqrt(2)]),
        ([[1, 2]], Scalarization.GEOM_MEAN, [1, 2], [np.power(4, 1 / 3)]),
    ],
)
def test_desirability_scalarization(values, scalarization, weights, expected):
    """The desirability scalarization yields the expected result."""
    actual = scalarize(values, scalarization, weights)
    assert np.array_equal(actual, expected), (expected, actual)
