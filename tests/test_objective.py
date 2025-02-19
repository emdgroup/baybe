"""Tests for the objective module."""

import numpy as np
import pandas as pd
import pytest
from cattrs import IterableValidationError

from baybe.objectives.desirability import DesirabilityObjective, scalarize
from baybe.objectives.enum import Scalarizer
from baybe.objectives.single import SingleTargetObjective
from baybe.parameters.numerical import NumericalContinuousParameter
from baybe.recommenders import BotorchRecommender
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
                scalarizer="FALSE_STUFF",
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
    ("values", "scalarizer", "weights", "expected"),
    [
        ([[1, 2]], Scalarizer.MEAN, [1, 1], [1.5]),
        ([[1, 2]], Scalarizer.MEAN, [1, 2], [5 / 3]),
        ([[1, 2]], Scalarizer.GEOM_MEAN, [1, 1], [np.sqrt(2)]),
        ([[1, 2]], Scalarizer.GEOM_MEAN, [1, 2], [np.power(4, 1 / 3)]),
    ],
)
def test_desirability_scalarization(values, scalarizer, weights, expected):
    """The desirability scalarization yields the expected result."""
    actual = scalarize(values, scalarizer, weights)
    assert np.array_equal(actual, expected), (expected, actual)


@pytest.mark.parametrize(
    ("mode", "bounds", "opt"),
    [("MIN", None, 0), ("MAX", None, 1), ("MIN", (0, 1), 0), ("MAX", (0, 1), 1)],
)
def test_single_objective(mode, bounds, opt):
    """Recommendations yield expected results with and without bounded objective."""
    searchspace = NumericalContinuousParameter("p", [0, 1]).to_searchspace()
    objective = NumericalTarget("t", mode=mode, bounds=bounds).to_objective()
    recommender = BotorchRecommender()
    measurements = pd.DataFrame(
        {"p": np.linspace(0, 1, 100), "t": np.linspace(0, 1, 100)}
    )
    rec = recommender.recommend(1, searchspace, objective, measurements)
    assert np.isclose(rec["p"].item(), opt)
