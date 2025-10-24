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
from baybe.symmetry import MirrorSymmetry
from baybe.utils.dataframe import create_fake_input


@pytest.mark.parametrize("mirror_aug", [True, False], ids=["mirror", "nomirror"])
@pytest.mark.parametrize("perm_aug", [True, False], ids=["perm", "noperm"])
@pytest.mark.parametrize("dep_aug", [True, False], ids=["dep", "nodep"])
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
            "Num_disc_2",
        ]
    ],
    ids=["p"],
)
def test_measurement_augmentation(
    parameters,
    surrogate_model,
    objective,
    constraints,
    dep_aug,
    perm_aug,
    mirror_aug,
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
        s_perm = c_perm.to_symmetry(perm_aug)
        s_dep = c_dep.to_symmetry(dep_aug)
        s_mirror = MirrorSymmetry("Num_disc_2", use_data_augmentation=mirror_aug)
        searchspace = SearchSpace.from_product(parameters, constraints)
        surrogate = evolve(surrogate_model, symmetries=[s_dep, s_perm, s_mirror])
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
        #  - n_mirror: 2 if the row is not on the mirror point, else 1
        #  - If augmentation is turned off, the corresponding factor becomes 1
        # We expect a given row to produce n_perm * (n_dep^k) * n_mirror points, where
        # k is the number of "Fraction_*" parameters having the "causing" value 0.0. The
        # total number of expected points after augmentation is the sum over the
        # expectations for all rows.
        dep_affected = [p for p in parameters if p.name in c_dep.affected_parameters[0]]
        n_dep = math.prod(len(p.active_values) for p in dep_affected) if dep_aug else 1
        n_perm = (  # number of permutations
            math.prod(range(1, len(c_perm.parameters) + 1)) if perm_aug else 1
        )
        n_expected = 0
        for _, row in measurements.iterrows():
            n_mirror = (
                2 if (mirror_aug and row["Num_disc_2"] != s_mirror.mirror_point) else 1
            )
            k = row[c_dep.parameters].eq(0).sum()
            n_expected += n_perm * np.pow(n_dep, k) * n_mirror

        # Check expectation
        if any([dep_aug, perm_aug, mirror_aug]):
            assert len(measurements_passed) == n_expected
        else:
            assert_frame_equal(measurements, measurements_passed)
