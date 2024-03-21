"""Test serialization of constraints."""

import hypothesis.strategies as st
import pytest
from hypothesis import given
from pytest import param

from baybe.constraints.base import Constraint

from ..hypothesis_strategies.constraints import (
    discrete_excludes_constraints,
    discrete_product_constraints,
    discrete_sum_constraints,
)


@pytest.mark.parametrize(
    "constraint_strategy",
    [
        param(discrete_excludes_constraints(), id="DiscreteExcludeConstraint"),
        param(discrete_sum_constraints(), id="DiscreteSumConstraint"),
        param(discrete_product_constraints(), id="DiscreteProductConstraint"),
    ],
)
@given(data=st.data())
def test_constraint_roundtrip(constraint_strategy, data):
    """A serialization roundtrip yields an equivalent object."""
    constraint = data.draw(constraint_strategy)
    string = constraint.to_json()
    constraint2 = Constraint.from_json(string)
    assert constraint == constraint2, (constraint, constraint2)


@pytest.mark.parametrize(
    "constraint_names",
    [[f"Constraint_{k}"] for k in range(1, 13)]  # Constraint_13 is expected to fail
    + [[f"ContiConstraint_{k}"] for k in range(1, 5)],
)
@pytest.mark.parametrize("n_grid_points", [5])
def test_constraint_serialization(constraints):
    constraint = constraints[0]
    string = constraint.to_json()
    constraint2 = Constraint.from_json(string)
    assert constraint == constraint2


@pytest.mark.parametrize(
    "constraint_names",
    [["Constraint_13"]],
)
@pytest.mark.parametrize("n_grid_points", [5])
def test_unsupported_serialization_of_custom_constraint(constraints):
    with pytest.raises(NotImplementedError):
        constraint = constraints[0]
        constraint.to_json()
