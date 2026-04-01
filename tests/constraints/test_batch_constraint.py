"""Tests for the discrete batch constraint."""

import pytest
from pytest import param

from baybe.constraints.discrete import DiscreteBatchConstraint
from baybe.exceptions import IncompatibilityError, InfeasibilityError
from baybe.parameters.numerical import NumericalDiscreteParameter
from baybe.recommenders import BotorchRecommender
from baybe.recommenders.pure.nonpredictive.sampling import (
    FPSRecommender,
    RandomRecommender,
)
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from baybe.utils.dataframe import create_fake_input

BATCH_SIZE = 3
TARGET = NumericalTarget("y")

_params = [
    NumericalDiscreteParameter("d0", values=(0.0, 0.5, 1.0)),
    NumericalDiscreteParameter("d1", values=(0.0, 0.5, 1.0)),
]


@pytest.mark.parametrize(
    ("constraints", "constrained_params", "batch_size"),
    [
        param(
            [DiscreteBatchConstraint(parameters=["d0"])],
            ["d0"],
            BATCH_SIZE,
            id="single",
        ),
        param(
            [
                DiscreteBatchConstraint(parameters=["d0"]),
                DiscreteBatchConstraint(parameters=["d1"]),
            ],
            ["d0", "d1"],
            1,
            id="multiple",
        ),
    ],
)
def test_batch_constraint_bayesian(constraints, constrained_params, batch_size):
    """BotorchRecommender respects batch constraints."""
    searchspace = SearchSpace.from_product(_params, constraints)
    measurements = create_fake_input(_params, [TARGET], n_rows=3)

    rec = BotorchRecommender().recommend(
        batch_size, searchspace, TARGET.to_objective(), measurements
    )
    assert rec.shape[0] == batch_size
    for p in constrained_params:
        assert rec[p].nunique() == 1


def test_batch_constraint_random_recommender():
    """RandomRecommender respects the batch constraint."""
    searchspace = SearchSpace.from_product(
        _params, [DiscreteBatchConstraint(parameters=["d0"])]
    )
    rec = RandomRecommender().recommend(BATCH_SIZE, searchspace)
    assert rec["d0"].nunique() == 1
    assert rec.shape[0] == BATCH_SIZE


def test_batch_constraint_unsupported_recommender():
    """Unsupported recommenders raise IncompatibilityError."""
    searchspace = SearchSpace.from_product(
        _params, [DiscreteBatchConstraint(parameters=["d0"])]
    )
    with pytest.raises(IncompatibilityError, match="does not support"):
        FPSRecommender().recommend(BATCH_SIZE, searchspace)


def test_batch_constraint_validation_multi_param():
    """DiscreteBatchConstraint requires exactly one parameter."""
    with pytest.raises(ValueError, match="exactly one parameter"):
        DiscreteBatchConstraint(parameters=["A", "B"])


def test_batch_constraint_validation_duplicate():
    """Two batch constraints on the same parameter are rejected."""
    constraints = [
        DiscreteBatchConstraint(parameters=["d0"]),
        DiscreteBatchConstraint(parameters=["d0"]),
    ]
    with pytest.raises(ValueError, match="same parameter"):
        SearchSpace.from_product(_params, constraints)


@pytest.mark.parametrize(
    ("constraints", "expected"),
    [
        param([], 0, id="none"),
        param([DiscreteBatchConstraint(parameters=["d0"])], 3, id="single"),
        param(
            [
                DiscreteBatchConstraint(parameters=["d0"]),
                DiscreteBatchConstraint(parameters=["d1"]),
            ],
            9,
            id="two",
        ),
    ],
)
def test_batch_constraint_n_theoretical_partitions(constraints, expected):
    """The n_theoretical_partitions property returns the correct count."""
    assert (
        SearchSpace.from_product(_params, constraints).discrete.n_theoretical_partitions
        == expected
    )


def test_batch_constraint_all_partitions_too_small():
    """All partitions infeasible raises InfeasibilityError."""
    searchspace = SearchSpace.from_product(
        _params, [DiscreteBatchConstraint(parameters=["d0"])]
    )
    measurements = create_fake_input(_params, [TARGET], n_rows=2)

    # Each d0 partition has 3 candidates, batch_size=4 exceeds all
    with pytest.raises(InfeasibilityError):
        BotorchRecommender().recommend(
            4, searchspace, TARGET.to_objective(), measurements
        )


@pytest.mark.parametrize(
    ("min_candidates", "expected_count"),
    [
        param(None, 3, id="no_filter"),
        param(4, 0, id="all_skipped"),
        param(3, 3, id="all_retained"),
    ],
)
def test_partition_masks_min_candidates(min_candidates, expected_count):
    """Partition mask filtering by min_candidates."""
    searchspace = SearchSpace.from_product(
        _params, [DiscreteBatchConstraint(parameters=["d0"])]
    )
    masks = list(
        searchspace.discrete.partition_masks(
            searchspace.discrete.exp_rep, min_candidates=min_candidates
        )
    )
    assert len(masks) == expected_count
