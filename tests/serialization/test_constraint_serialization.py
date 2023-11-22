"""Test serialization of constraints."""

import pytest

from baybe.constraints.base import Constraint


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
