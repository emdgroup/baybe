"""Tests for augmentation of measurements."""

import math
from unittest.mock import patch

import numpy as np
import pytest
from attrs import evolve
from pandas.testing import assert_frame_equal

from baybe.acquisition import qLogEI
from baybe.constraints import (
    DiscretePermutationInvarianceConstraint,
)
from baybe.recommenders import BotorchRecommender
from baybe.searchspace import SearchSpace
from baybe.utils.dataframe import create_fake_input


@pytest.mark.parametrize(
    "perm_augmentation", [True, False], ids=["perm_aug", "perm_noaug"]
)
@pytest.mark.parametrize(
    "dep_augmentation", [True, False], ids=["dep_aug", "dep_noaug"]
)
@pytest.mark.parametrize(
    "constraint_names", [["Constraint_11", "Constraint_7"]], ids=["c"]
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
        c_perm = next(
            c
            for c in constraints
            if isinstance(c, DiscretePermutationInvarianceConstraint)
        )
        c_dep = c_perm.dependencies
        s_perm = c_perm.to_symmetry(perm_augmentation)
        s_dep = c_dep.to_symmetry(dep_augmentation)
        searchspace = SearchSpace.from_product(parameters, constraints)
        surrogate = evolve(surrogate_model, symmetries=[s_dep, s_perm])
        recommender = BotorchRecommender(
            surrogate_model=surrogate, acquisition_function=qLogEI()
        )

        # Perform call and watch measurements
        measurements = create_fake_input(parameters, objective.targets, 5)
        recommender.recommend(1, searchspace, objective, measurements)
        measurements_passed = called_args_list[0][0][3]  # take 4th arg from first call

        # Create expectation
        # We calculate how many degenerate points the augmentation should create:
        #  - n_dep: Product of the number of active values for all affected parameters
        #  - n_perm: Number of permutations possible
        #  - If augmentation is turned off, the corresponding factor becomes 1
        # We expect a given row to produce n_perm * (n_dep^k) points, where k is the
        # number of "Fraction_*" parameters having the "causing" value 0.0. The total
        # number of expected points after augmentation is the sum over the expectations
        # for all rows.
        dep_affected = [p for p in parameters if p.name in c_dep.affected_parameters[0]]
        n_dep = (
            math.prod(len(p.active_values) for p in dep_affected)
            if dep_augmentation
            else 1
        )
        n_perm = (  # number of permutations
            math.prod(range(1, len(c_perm.parameters) + 1)) if perm_augmentation else 1
        )
        n_expected = 0
        for _, row in measurements.iterrows():
            n_expected += n_perm * np.pow(n_dep, row[c_dep.parameters].eq(0).sum())

        # Check expectation
        if any([dep_augmentation, perm_augmentation]):
            assert len(measurements_passed) == n_expected
        else:
            assert_frame_equal(measurements, measurements_passed)
