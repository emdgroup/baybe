"""Tests for the objective module."""

import numpy as np
import pandas as pd
import pytest
from cattrs import IterableValidationError

from baybe.objectives.desirability import DesirabilityObjective
from baybe.objectives.enum import Scalarizer
from baybe.objectives.single import SingleTargetObjective
from baybe.parameters.numerical import NumericalContinuousParameter
from baybe.recommenders import BotorchRecommender
from baybe.targets import NumericalTarget
from baybe.targets.transformation import ClampingTransformation


class TestInvalidObjectiveCreation:
    """Invalid objective creation raises expected error."""

    # Two example targets used in the tests
    two_targets = [
        NumericalTarget.ramp(
            name="Target_1",
            cutoffs=(0, 100),
        ),
        NumericalTarget.ramp(
            name="Target_2",
            cutoffs=(0, 100),
            descending=True,
        ),
    ]

    def test_empty_target_list(self):
        with pytest.raises(ValueError):
            DesirabilityObjective(targets=[])

    def test_wrong_target_type(self):
        with pytest.raises(TypeError):
            SingleTargetObjective(target={"A": 1, "B": 2})

    def test_negative_targets_for_desirability(self):
        t1 = NumericalTarget("t1").clamp(0, 1)

        # Is normalized but the minimize flag appends another transformation
        t2 = NumericalTarget(
            "t2", transformation=ClampingTransformation(0, 1), minimize=True
        )

        with pytest.raises(ValueError, match="transformed to a non-negative range"):
            DesirabilityObjective([t1, t2])

    def test_unnormalized_targets_for_desirability(self):
        """Unnormalized targets are not allowed unless explicitly declared."""
        t1 = NumericalTarget("t1").clamp(min=1, max=2)
        t2 = NumericalTarget("t2").clamp(min=0, max=3)
        with pytest.raises(ValueError, match="are not normalized"):
            DesirabilityObjective([t1, t2])
        DesirabilityObjective([t1, t2], require_normalization=False)
        DesirabilityObjective([t1.normalize(), t2.normalize()])

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
        ([[1, 2]], Scalarizer.MEAN, [1, 1], 1.5),
        ([[1, 2]], Scalarizer.MEAN, [1, 2], 5 / 3),
        ([[1, 2]], Scalarizer.GEOM_MEAN, [1, 1], np.sqrt(2)),
        ([[1, 2]], Scalarizer.GEOM_MEAN, [1, 2], np.power(4, 1 / 3)),
    ],
    ids=["mean1", "mean2", "geom1", "geom2"],
)
def test_desirability_scalarization(values, scalarizer, weights, expected):
    """The desirability scalarization yields the expected result."""
    obj = DesirabilityObjective(
        [
            NumericalTarget("t1").clamp(min=0),
            NumericalTarget("t2").clamp(min=0),
        ],
        weights,
        scalarizer,
        require_normalization=False,
    )
    actual = obj.transform(pd.DataFrame(values, columns=["t1", "t2"])).values.item()
    assert np.isclose(actual, expected), (expected, actual)


@pytest.mark.parametrize(
    ("target", "opt"),
    [
        (NumericalTarget("t", minimize=True), 0),
        (NumericalTarget("t"), 1),
        (NumericalTarget.ramp("t", cutoffs=(0, 1), descending=True), 0),
        (NumericalTarget.ramp("t", cutoffs=(0, 1)), 1),
    ],
)
def test_single_objective(target, opt):
    """Recommendations yield expected results with and without bounded objective."""
    searchspace = NumericalContinuousParameter("p", [0, 1]).to_searchspace()
    objective = target.to_objective()
    recommender = BotorchRecommender()
    measurements = pd.DataFrame(
        {"p": np.linspace(0, 1, 100), "t": np.linspace(0, 1, 100)}
    )
    rec = recommender.recommend(1, searchspace, objective, measurements)
    assert np.isclose(rec["p"].item(), opt)
