"""Tests for the discrete linear constraint."""

import pytest
from pytest import param

from baybe.constraints.discrete import DiscreteLinearConstraint
from tests.constraints.discrete.utils import (
    assert_filter,
    assert_partial_rejected,
    requires_fulltest,
)

pytestmark = requires_fulltest


_LINEAR_DATA = [
    {"amount_1": 0.0, "amount_2": 0.0, "solvent": "water"},
    {"amount_1": 1.0, "amount_2": 0.0, "solvent": "ethanol"},
    {"amount_1": 1.0, "amount_2": 1.0, "solvent": "methanol"},
    {"amount_1": 2.0, "amount_2": 1.0, "solvent": "acetone"},
    {"amount_1": 5.0, "amount_2": 0.0, "solvent": "water"},
    {"amount_1": 10.0, "amount_2": 0.0, "solvent": "hexane"},
]


@pytest.mark.parametrize(
    ("coefficients", "operator", "rhs", "tolerance", "exclude", "expected_rows"),
    [
        param(
            None,
            "<",
            2.0,
            None,
            False,
            [
                {"amount_1": 0.0, "amount_2": 0.0, "solvent": "water"},
                {"amount_1": 1.0, "amount_2": 0.0, "solvent": "ethanol"},
            ],
            id="operator-lt",
        ),
        param(
            None,
            "<=",
            2.0,
            None,
            False,
            [
                {"amount_1": 0.0, "amount_2": 0.0, "solvent": "water"},
                {"amount_1": 1.0, "amount_2": 0.0, "solvent": "ethanol"},
                {"amount_1": 1.0, "amount_2": 1.0, "solvent": "methanol"},
            ],
            id="operator-le",
        ),
        param(
            None,
            "=",
            2.0,
            None,
            False,
            [{"amount_1": 1.0, "amount_2": 1.0, "solvent": "methanol"}],
            id="operator-eq",
        ),
        param(
            None,
            "==",
            2.0,
            None,
            False,
            [{"amount_1": 1.0, "amount_2": 1.0, "solvent": "methanol"}],
            id="operator-eq-alias",
        ),
        param(
            None,
            "!=",
            2.0,
            None,
            False,
            [
                {"amount_1": 0.0, "amount_2": 0.0, "solvent": "water"},
                {"amount_1": 1.0, "amount_2": 0.0, "solvent": "ethanol"},
                {"amount_1": 2.0, "amount_2": 1.0, "solvent": "acetone"},
                {"amount_1": 5.0, "amount_2": 0.0, "solvent": "water"},
                {"amount_1": 10.0, "amount_2": 0.0, "solvent": "hexane"},
            ],
            id="operator-ne",
        ),
        param(
            None,
            ">",
            2.0,
            None,
            False,
            [
                {"amount_1": 2.0, "amount_2": 1.0, "solvent": "acetone"},
                {"amount_1": 5.0, "amount_2": 0.0, "solvent": "water"},
                {"amount_1": 10.0, "amount_2": 0.0, "solvent": "hexane"},
            ],
            id="operator-gt",
        ),
        param(
            None,
            ">=",
            2.0,
            None,
            False,
            [
                {"amount_1": 1.0, "amount_2": 1.0, "solvent": "methanol"},
                {"amount_1": 2.0, "amount_2": 1.0, "solvent": "acetone"},
                {"amount_1": 5.0, "amount_2": 0.0, "solvent": "water"},
                {"amount_1": 10.0, "amount_2": 0.0, "solvent": "hexane"},
            ],
            id="operator-ge",
        ),
        param(
            (2.0, 1.0),
            "<=",
            2.0,
            None,
            False,
            [
                {"amount_1": 0.0, "amount_2": 0.0, "solvent": "water"},
                {"amount_1": 1.0, "amount_2": 0.0, "solvent": "ethanol"},
            ],
            id="coefficients-scaled",
        ),
        param(
            (1.0, -1.0),
            "<=",
            0.0,
            None,
            False,
            [
                {"amount_1": 0.0, "amount_2": 0.0, "solvent": "water"},
                {"amount_1": 1.0, "amount_2": 1.0, "solvent": "methanol"},
            ],
            id="coefficients-negative",
        ),
        param(
            (2.0, 1.0),
            "=",
            3.0,
            None,
            False,
            [{"amount_1": 1.0, "amount_2": 1.0, "solvent": "methanol"}],
            id="coefficients-equality",
        ),
        param(
            None,
            "=",
            2.5,
            0.5,
            False,
            [
                {"amount_1": 1.0, "amount_2": 1.0, "solvent": "methanol"},
                {"amount_1": 2.0, "amount_2": 1.0, "solvent": "acetone"},
            ],
            id="tolerance",
        ),
        param(
            None,
            "<=",
            2.0,
            None,
            True,
            [
                {"amount_1": 2.0, "amount_2": 1.0, "solvent": "acetone"},
                {"amount_1": 5.0, "amount_2": 0.0, "solvent": "water"},
                {"amount_1": 10.0, "amount_2": 0.0, "solvent": "hexane"},
            ],
            id="exclude",
        ),
    ],
)
def test_linear_constraint(
    coefficients,
    operator,
    rhs,
    tolerance,
    exclude,
    expected_rows,
    backend,
):
    """Cover all operator, coefficient, tolerance, and exclusion regimes."""
    kwargs = {} if coefficients is None else {"coefficients": coefficients}
    constraint = DiscreteLinearConstraint(
        parameters=["amount_1", "amount_2"],
        operator=operator,
        rhs=rhs,
        tolerance=tolerance,
        exclude=exclude,
        **kwargs,
    )
    assert_filter(constraint, _LINEAR_DATA, expected_rows, backend)


def test_linear_constraint_partial_rejected(backend):
    """Linear constraints defer until all parameters are available."""
    constraint = DiscreteLinearConstraint(
        parameters=["amount_1", "amount_2"],
        operator="<=",
        rhs=2.0,
    )
    assert_partial_rejected(constraint, [{"amount_1": 1.0}], backend)
