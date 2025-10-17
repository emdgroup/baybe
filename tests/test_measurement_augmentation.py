"""Tests for augmentation of measurements."""

from unittest.mock import patch

import pytest
from attrs import evolve
from pandas.testing import assert_frame_equal

from baybe.acquisition import qLogEI
from baybe.recommenders import BotorchRecommender
from baybe.searchspace import SearchSpace
from baybe.symmetry import DependencySymmetry, PermutationSymmetry
from baybe.utils.dataframe import create_fake_input


@pytest.mark.parametrize(
    "perm_augmentation", [True, False], ids=["perm_aug", "perm_noaug"]
)
@pytest.mark.parametrize(
    "dep_augmentation", [True, False], ids=["dep_aug", "dep_noaug"]
)
@pytest.mark.parametrize(
    "constraint_names", [["Constraint_2", "Constraint_11"]], ids=["c"]
)  # constraints are not strictly needed for this test but reduce runtime
@pytest.mark.parametrize(
    "parameter_names",
    [
        [
            "Solvent_1",
            "Solvent_2",
            "Solvent_3",
            "Fraction_1",
            "Fraction_2",
            "Fraction_3",
            "Switch_1",
        ]
    ],
    ids=["p"],
)
def test_measurement_augmentation(
    parameters,
    surrogate_model,
    objective,
    constraints,
    dep_augmentation,
    perm_augmentation,
):
    """Measurement augmentation is performed if configured."""
    original_to_botorch = qLogEI.to_botorch
    called_args_list = []

    def spy(self, *args, **kwargs):
        called_args_list.append((args, kwargs))
        return original_to_botorch(self, *args, **kwargs)

    with patch.object(qLogEI, "to_botorch", side_effect=spy, autospec=True):
        # Basic setup
        s1 = DependencySymmetry(
            parameters=constraints[0].parameters,
            conditions=constraints[0].conditions,
            affected_parameters=constraints[0].affected_parameters,
            use_data_augmentation=dep_augmentation,
        )
        s2 = PermutationSymmetry(
            parameters=constraints[1].parameters,
            use_data_augmentation=perm_augmentation,
        )
        searchspace = SearchSpace.from_product(parameters, constraints)
        surrogate = evolve(surrogate_model, symmetries=[s1, s2])
        acqf = qLogEI()
        recommender = BotorchRecommender(
            surrogate_model=surrogate, acquisition_function=acqf
        )

        # Create measurements which would be augmented
        measurements = create_fake_input(parameters, objective.targets, 4)
        measurements.loc[:, "Switch_1"] = ["on", "on", "off", "off"]
        measurements.loc[:, "Fraction_1"] = [0.0, 100.0, 0.0, 100.0]

        # Perform call
        recommender.recommend(2, searchspace, objective, measurements)
        measurements_passed = called_args_list[0][0][3]  # take 4th arg from first call

        # Check expectation
        # If any of the symmetries consider augmentation, the measurements passed to
        # `to_botorch` should be larger than the original measurements - otherwise,
        # they should be identical
        len1 = len(measurements)
        len2 = len(measurements_passed)
        if any([dep_augmentation, perm_augmentation]):
            assert len1 < len2, (len1, len2)
        else:
            assert_frame_equal(measurements, measurements_passed)
