"""Tests for the objective module."""

import pytest

from baybe.objective import Objective
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
        with pytest.raises(TypeError):
            Objective(
                mode="DESIRABILITY",
                weights=[1, "ABC"],
                targets=self.two_targets,
            )

    def test_wrong_weights_type(self):
        with pytest.raises(TypeError):
            Objective(
                mode="DESIRABILITY",
                weights="ABC",
                targets=self.two_targets,
            )
