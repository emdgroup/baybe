"""Tests for augmentation of measurements."""

import inspect
from unittest import mock
from unittest.mock import patch

import pytest
from attrs import evolve
from pandas.testing import assert_frame_equal
from pytest import param

from baybe.acquisition import qLogEI
from baybe.recommenders import BotorchRecommender
from baybe.searchspace import SearchSpace
from baybe.utils.dataframe import create_fake_input


def spy_mock(instance):
    members = inspect.getmembers(instance, inspect.ismethod)
    attrs = {"%s.side_effect" % k: v for k, v in members}
    return mock.Mock(**attrs)


@pytest.mark.parametrize("surrogate_considers", [True, False])
@pytest.mark.parametrize(
    "constraints_consider",
    [
        param([True, False], id="dep"),
        param([False, True], id="perm"),
        param([True, True], id="dep+perm"),
        param([False, False], id="none"),
    ],
)
@pytest.mark.parametrize(
    "constraint_names", [["Constraint_2", "Constraint_11"]], ids=["c"]
)
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
    constraints,
    surrogate_model,
    objective,
    constraints_consider,
    surrogate_considers,
):
    """Measurement augmentation is performed if configured."""
    original_to_botorch = qLogEI.to_botorch
    called_args_list = []

    def spy_side_effect(self, *args, **kwargs):
        called_args_list.append((args, kwargs))
        return original_to_botorch(self, *args, **kwargs)

    with patch.object(qLogEI, "to_botorch", side_effect=spy_side_effect, autospec=True):
        # Basic setup
        c1 = evolve(constraints[0], consider_data_augmentation=constraints_consider[0])
        c2 = evolve(constraints[1], consider_data_augmentation=constraints_consider[1])
        searchspace = SearchSpace.from_product(parameters, [c1, c2])
        surrogate = evolve(
            surrogate_model, consider_data_augmentation=surrogate_considers
        )
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
        # If the surrogate considers augmentation and any of the constraints consider
        # augmentation, the measurements passed to `to_botorch` should be larger than
        # the original measurements - otherwise, they should be identical
        len1 = len(measurements)
        len2 = len(measurements_passed)
        if any(constraints_consider) and surrogate_considers:
            assert len1 < len2, (len1, len2)
        else:
            assert_frame_equal(measurements, measurements_passed)
