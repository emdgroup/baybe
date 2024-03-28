"""Tests for the objective module."""

import numpy as np
import pytest
from cattrs import IterableValidationError

from baybe.objective import Objective
from baybe.objectives.desirability import scalarize
from baybe.objectives.enum import CombineFunc
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
            Objective(
                mode="SINGLE",
                targets=[],
            )

    def test_wrong_target_type(self):
        with pytest.raises(ValueError):
            Objective(
                mode="SINGLE",
                targets={"A": 1, "B": 2},
            )

    def test_missing_bounds_for_desirability(self):
        with pytest.raises(ValueError):
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

    def test_invalid_combination_function(self):
        with pytest.raises(ValueError):
            Objective(
                mode="DESIRABILITY",
                combine_func="FALSE_STUFF",
                targets=self.two_targets,
            )

    def test_wrong_number_of_weights(self):
        with pytest.raises(ValueError):
            Objective(
                mode="DESIRABILITY",
                weights=[1, 2, 3],
                targets=self.two_targets,
            )

    def test_non_numeric_weights(self):
        with pytest.raises(IterableValidationError):
            Objective(
                mode="DESIRABILITY",
                weights=[1, "ABC"],
                targets=self.two_targets,
            )

    def test_wrong_weights_type(self):
        with pytest.raises(IterableValidationError):
            Objective(
                mode="DESIRABILITY",
                weights="ABC",
                targets=self.two_targets,
            )


@pytest.mark.parametrize(
    ("values", "combine_func", "weights", "expected"),
    [
        ([[1, 2]], CombineFunc.MEAN, [1, 1], [1.5]),
        ([[1, 2]], CombineFunc.MEAN, [1, 2], [5 / 3]),
        ([[1, 2]], CombineFunc.GEOM_MEAN, [1, 1], [np.sqrt(2)]),
        ([[1, 2]], CombineFunc.GEOM_MEAN, [1, 2], [np.power(4, 1 / 3)]),
    ],
)
def test_desirability_scalarization(values, combine_func, weights, expected):
    """The desirability scalarization yields the expected result."""
    actual = scalarize(values, combine_func, weights)
    assert np.array_equal(actual, expected), (expected, actual)
