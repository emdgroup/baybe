"""Test serialization of constraints."""

import pytest
from hypothesis import given

from baybe.constraints.base import Constraint

from ..hypothesis_strategies.constraints import discrete_excludes_constraints


@given(discrete_excludes_constraints())
def test_constraint_roundtrip(constraint):
    """A serialization roundtrip yields an equivalent object."""
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
