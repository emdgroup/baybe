"""Tests for partitioning constraints in hybrid search spaces."""

import pytest
from pytest import param

from baybe.constraints.continuous import ContinuousCardinalityConstraint
from baybe.constraints.discrete import (
    DiscreteBatchConstraint,
    DiscreteCardinalityConstraint,
)
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
TARGET = NumericalTarget("t")

_discrete_params = [
    NumericalDiscreteParameter("d0", values=(0.0, 0.5, 1.0)),
    NumericalDiscreteParameter("d1", values=(0.0, 0.5, 1.0)),
]
_continuous_params = [
    NumericalContinuousParameter("c0", bounds=(0, 1)),
    NumericalContinuousParameter("c1", bounds=(0, 1)),
]
_all_params = [*_discrete_params, *_continuous_params]


@pytest.mark.parametrize(
    "constraints",
    [
        param(
            [
                ContinuousCardinalityConstraint(
                    parameters=["c0", "c1"], max_cardinality=MAX_CARDINALITY
                )
            ],
            id="continuous_cardinality",
        ),
        param(
            [
                DiscreteCardinalityConstraint(
                    parameters=["d0", "d1"], max_cardinality=MAX_CARDINALITY
                )
            ],
            id="discrete_cardinality",
        ),
        param(
            [
                DiscreteCardinalityConstraint(
                    parameters=["d0", "d1"], max_cardinality=MAX_CARDINALITY
                ),
                ContinuousCardinalityConstraint(
                    parameters=["c0", "c1"], max_cardinality=MAX_CARDINALITY
                ),
            ],
            id="both_cardinality",
        ),
        param(
            [DiscreteBatchConstraint(parameters=["d0"])],
            id="batch_single",
        ),
        param(
            [
                DiscreteBatchConstraint(parameters=["d0"]),
                DiscreteBatchConstraint(parameters=["d1"]),
            ],
            id="batch_multiple",
        ),
        param(
            [
                DiscreteBatchConstraint(parameters=["d0"]),
                ContinuousCardinalityConstraint(
                    parameters=["c0", "c1"], max_cardinality=MAX_CARDINALITY
                ),
            ],
            id="batch_and_cardinality",
        ),
    ],
)
def test_partition_constraints_hybrid(constraints):
    """Partitioning constraints are respected in hybrid search spaces."""
    searchspace = SearchSpace.from_product(_all_params, constraints)
    measurements = create_fake_input(_all_params, [TARGET], n_rows=3)

    rec = BotorchRecommender().recommend(
        BATCH_SIZE, searchspace, TARGET.to_objective(), measurements
    )

    for c in constraints:
        if isinstance(c, ContinuousCardinalityConstraint):
            assert is_cardinality_fulfilled(
                rec, searchspace.continuous, check_minimum=False
            )
        elif isinstance(c, DiscreteCardinalityConstraint):
            n_nonzero = (rec[list(c.parameters)] != 0.0).sum(axis=1)
            assert (n_nonzero <= c.max_cardinality).all(), (
                f"Discrete cardinality constraint violated: {n_nonzero.tolist()}"
            )
        elif isinstance(c, DiscreteBatchConstraint):
            assert rec[c.parameters[0]].nunique() == 1, (
                f"Batch constraint violated for '{c.parameters[0]}': "
                f"found {rec[c.parameters[0]].nunique()} unique values"
            )
