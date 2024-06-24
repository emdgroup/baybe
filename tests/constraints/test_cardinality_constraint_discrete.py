"""Test for the discrete cardinality constraint."""

from itertools import combinations

import numpy as np
import pytest

from baybe.constraints.discrete import DiscreteCardinalityConstraint
from baybe.parameters.numerical import NumericalDiscreteParameter
from baybe.searchspace import SearchSpace

# Combinations of cardinalities to be tested
N_PARAMETERS = 5
_cardinalities = sorted(combinations(range(0, N_PARAMETERS), 2))


@pytest.mark.parametrize(
    "cardinalities", _cardinalities, ids=[str(x) for x in _cardinalities]
)
def test_cardinality_constraint_discrete(cardinalities, n_grid_points):
    """Elements of a unit-tube subspace under a cardinality constraint have the
    expected number of non-zeros.
    """  # noqa
    # cardinality numbers
    min_cardinality, max_cardinality = cardinalities

    # Parameters
    parameters = [
        NumericalDiscreteParameter(
            name=f"x_{i}", values=tuple(np.linspace(0, 1, n_grid_points))
        )
        for i in range(N_PARAMETERS)
    ]

    # Constraint
    constraint = [
        DiscreteCardinalityConstraint(
            parameters=[f"x_{i}" for i in range(N_PARAMETERS)],
            min_cardinality=min_cardinality,
            max_cardinality=max_cardinality,
        )
    ]

    # Searchspace
    searchspace = SearchSpace.from_product(parameters, constraint)

    # Assert that cardinality constraint is fulfilled
    assert (
        (searchspace.discrete.exp_rep != 0.0)
        .sum(axis=1)
        .between(min_cardinality, max_cardinality)
        .all()
    )
