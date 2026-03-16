"""Tests for cardinality constraints in hybrid search spaces."""

import pytest

from baybe.constraints.continuous import ContinuousCardinalityConstraint
from baybe.constraints.discrete import DiscreteCardinalityConstraint
from baybe.constraints.utils import is_cardinality_fulfilled
from baybe.parameters.numerical import (
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.recommenders import BotorchRecommender
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from baybe.utils.dataframe import create_fake_input

BATCH_SIZE = 2
MAX_CARDINALITY = 1

_discrete_params = [
    NumericalDiscreteParameter(f"d{i}", values=(0.0, 0.5, 1.0)) for i in range(2)
]
_continuous_params = [
    NumericalContinuousParameter(f"c{i}", bounds=(0, 1)) for i in range(2)
]


@pytest.mark.parametrize(
    ("disc_params", "conti_params", "constraints"),
    [
        pytest.param(
            [NumericalDiscreteParameter("d", values=(0.0, 1.0))],
            _continuous_params,
            [
                ContinuousCardinalityConstraint(
                    parameters=[p.name for p in _continuous_params],
                    max_cardinality=MAX_CARDINALITY,
                )
            ],
            id="conti",
        ),
        pytest.param(
            _discrete_params,
            [NumericalContinuousParameter("c", bounds=(0, 1))],
            [
                DiscreteCardinalityConstraint(
                    parameters=[p.name for p in _discrete_params],
                    max_cardinality=MAX_CARDINALITY,
                )
            ],
            id="disc",
        ),
        pytest.param(
            _discrete_params,
            _continuous_params,
            [
                DiscreteCardinalityConstraint(
                    parameters=[p.name for p in _discrete_params],
                    max_cardinality=MAX_CARDINALITY,
                ),
                ContinuousCardinalityConstraint(
                    parameters=[p.name for p in _continuous_params],
                    max_cardinality=MAX_CARDINALITY,
                ),
            ],
            id="hybrid",
        ),
    ],
)
def test_cardinality_constraint_hybrid(disc_params, conti_params, constraints):
    """Cardinality constraints are respected in hybrid search spaces."""
    parameters = [*disc_params, *conti_params]
    searchspace = SearchSpace.from_product(parameters, constraints)
    target = NumericalTarget("t")
    measurements = create_fake_input(parameters, [target])

    rec = BotorchRecommender().recommend(
        BATCH_SIZE, searchspace, target.to_objective(), measurements
    )

    for c in constraints:
        if isinstance(c, ContinuousCardinalityConstraint):
            assert is_cardinality_fulfilled(
                rec, searchspace.continuous, check_minimum=False
            )
        elif isinstance(c, DiscreteCardinalityConstraint):
            n_nonzero = (rec[list(c.parameters)] != 0.0).sum(axis=1)
            assert (n_nonzero <= c.max_cardinality).all()
