"""Test for the discrete cardinality constraint."""

from itertools import combinations

import numpy as np
import pytest

from baybe.constraints.discrete import DiscreteCardinalityConstraint
from baybe.parameters.numerical import NumericalDiscreteParameter
from baybe.searchspace import SearchSpace

# Combinations of cardinalities to be tested
cardinality_bounds_combinations = sorted(combinations(range(0, 5), 2))


@pytest.mark.parametrize(
    "cardinality_bounds",
    cardinality_bounds_combinations,
    ids=[str(x) for x in cardinality_bounds_combinations],
)
def test_cardinality_constraint_discrete(
    cardinality_bounds: tuple[int, int], n_grid_points: int
):
    """Elements of a (discretized) unit-cube subspace under a cardinality constraint
    have the expected number of non-zeros."""  # noqa

    N_PARAMETERS = 5
    min_cardinality, max_cardinality = cardinality_bounds

    parameters = [
        NumericalDiscreteParameter(
            name=f"x_{i}", values=tuple(np.linspace(0, 1, n_grid_points))
        )
        for i in range(N_PARAMETERS)
    ]

    constraint = [
        DiscreteCardinalityConstraint(
            parameters=[f"x_{i}" for i in range(N_PARAMETERS)],
            min_cardinality=min_cardinality,
            max_cardinality=max_cardinality,
        )
    ]

    searchspace = SearchSpace.from_product(parameters, constraint)

    # Assert that cardinality constraint is fulfilled
    assert (
        (searchspace.discrete.exp_rep != 0.0)
        .sum(axis=1)
        .between(min_cardinality, max_cardinality)
        .all()
    )
