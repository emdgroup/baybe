# pylint: disable=missing-module-docstring, missing-function-docstring

import pytest

from baybe.constraints import Constraint


@pytest.mark.parametrize(
    "constraint_names",
    [[f"Constraint_{k}"] for k in range(1, 13)],
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
def test_unsupported_constraints(constraints):
    with pytest.raises(NotImplementedError):
        constraint = constraints[0]
        string = constraint.to_json()
        constraint2 = Constraint.from_json(string)
        assert constraint == constraint2
