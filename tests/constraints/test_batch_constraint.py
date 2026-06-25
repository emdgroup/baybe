"""Tests for the discrete batch constraint."""

from contextlib import nullcontext

import pytest
from pytest import param

from baybe.constraints import DiscreteExcludeConstraint, SubSelectionCondition
from baybe.constraints.discrete import DiscreteBatchConstraint
from baybe.exceptions import (
    IncompatibilityError,
    InfeasibilityError,
    UnusedObjectWarning,
)
from baybe.parameters.numerical import NumericalDiscreteParameter
from baybe.recommenders import BotorchRecommender
from baybe.recommenders.pure.nonpredictive.base import NonPredictiveRecommender
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
    "recommender",
    [
        param(BotorchRecommender(), id="botorch"),
        param(RandomRecommender(), id="random"),
    ],
)
@pytest.mark.parametrize(
    ("constrained_params", "batch_size"),
    [
        param(["d0"], BATCH_SIZE, id="single"),
        param(["d0", "d1"], 1, id="multiple"),
    ],
)
def test_batch_constraint(constrained_params, batch_size, recommender):
    """Recommenders respecting batch constraints return uniform batches."""
    constraints = [DiscreteBatchConstraint(parameters=[p]) for p in constrained_params]
    searchspace = SearchSpace.from_product(_params, constraints)
    measurements = create_fake_input(_params, [TARGET], n_rows=3)
    ctx = (
        pytest.warns(UnusedObjectWarning)
        if isinstance(recommender, NonPredictiveRecommender)
        else nullcontext()
    )
    with ctx:
        rec = recommender.recommend(
            batch_size, searchspace, TARGET.to_objective(), measurements
        )
    assert rec.shape[0] == batch_size
    for p in constrained_params:
        assert rec[p].nunique() == 1


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
def test_batch_constraint_n_subsets(constraints, expected):
    """The n_subsets property returns the correct count."""
    assert SearchSpace.from_product(_params, constraints).discrete.n_subsets == expected


def test_batch_constraint_all_subsets_too_small():
    """All subsets infeasible raises InfeasibilityError."""
    searchspace = SearchSpace.from_product(
        _params, [DiscreteBatchConstraint(parameters=["d0"])]
    )
    measurements = create_fake_input(_params, [TARGET], n_rows=2)

    # Each d0 subset has 3 candidates, batch_size=4 exceeds all
    with pytest.raises(InfeasibilityError):
        BotorchRecommender().recommend(
            4, searchspace, TARGET.to_objective(), measurements
        )


@pytest.mark.parametrize(
    ("min_candidates", "expected_count", "constraint"),
    [
        param(None, 3, None, id="no_filter"),
        param(4, 0, None, id="all_skipped"),
        param(3, 3, None, id="all_retained"),
        param(
            2,
            2,
            DiscreteExcludeConstraint(
                parameters=["d0", "d1"],
                conditions=[
                    SubSelectionCondition(selection=[0.0]),
                    SubSelectionCondition(selection=[0.0, 0.5]),
                ],
            ),
            id="partial",
        ),
    ],
)
def test_subset_masks_min_candidates(min_candidates, expected_count, constraint):
    """Subset mask filtering by min_candidates."""
    constraints = [DiscreteBatchConstraint(parameters=["d0"])]
    if constraint is not None:
        constraints.append(constraint)
    searchspace = SearchSpace.from_product(_params, constraints)
    masks = list(searchspace.discrete.subset_masks(min_candidates=min_candidates))
    assert len(masks) == expected_count
