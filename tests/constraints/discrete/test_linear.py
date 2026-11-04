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

##### API #####


@pytest.mark.parametrize(
    ("parameters", "error", "match"),
    [
        param([], ValueError, "Length of 'parameters' must be >= 1", id="empty"),
        param(["A", "A"], ValueError, "must have unique values", id="duplicate"),
        param([1, 2], TypeError, "must be <class 'str'>", id="type-int"),
        param(["A", None], TypeError, "must be <class 'str'>", id="type-none"),
    ],
)
def test_linear_constraint_invalid_parameters(parameters, error, match):
    """Linear constraints require a non-empty list of unique parameter names."""
    with pytest.raises(error, match=match):
        DiscreteLinearConstraint(
            parameters=parameters,
            operator="<=",
            rhs=1.0,
        )


@pytest.mark.parametrize(
    "operator",
    [
        param("~", id="unknown"),
        param("<>", id="wrong"),
        param("=<", id="reversed"),
    ],
)
def test_linear_constraint_invalid_operator(operator):
    """Linear constraints reject unknown comparison operators."""
    with pytest.raises(ValueError, match="'operator' must be in"):
        DiscreteLinearConstraint(parameters=["A", "B"], operator=operator, rhs=1.0)


@pytest.mark.parametrize(
    ("coefficients", "match"),
    [
        param((1.0, 2.0), "'coefficients' list must have one", id="length-mismatch"),
        param((1.0, 0.0, 1.0), "'coefficients' must be non-zero", id="zero"),
        param((1.0, float("nan"), 1.0), "cannot be 'nan'", id="nan"),
        param((1.0, float("inf"), 1.0), "cannot be 'inf'", id="inf"),
    ],
)
def test_linear_constraint_invalid_coefficients(coefficients, match):
    """Linear constraints reject invalid coefficient vectors."""
    with pytest.raises(ValueError, match=match):
        DiscreteLinearConstraint(
            parameters=["A", "B", "C"],
            operator="<=",
            rhs=1.0,
            coefficients=coefficients,
        )


@pytest.mark.parametrize(
    ("rhs", "match"),
    [
        param(float("nan"), "cannot be 'nan'", id="nan"),
        param(float("inf"), "cannot be 'inf'", id="inf"),
        param(-float("inf"), "cannot be 'inf'", id="neg-inf"),
    ],
)
def test_linear_constraint_invalid_rhs(rhs, match):
    """Linear constraints reject non-finite right-hand sides."""
    with pytest.raises(ValueError, match=match):
        DiscreteLinearConstraint(parameters=["A", "B"], operator="<=", rhs=rhs)


@pytest.mark.parametrize(
    ("operator", "tolerance", "match"),
    [
        param("=", float("nan"), "cannot be 'nan'", id="nan"),
        param("=", float("inf"), "cannot be 'inf'", id="inf"),
        param("=", -float("inf"), "cannot be 'inf'", id="neg-inf"),
        param("=", 0.0, "must be > 0", id="zero"),
        param("=", -1.0, "must be > 0", id="negative"),
        param(">=", 0.1, "valid with the following operators", id="wrong-operator"),
    ],
)
def test_linear_constraint_invalid_tolerance(operator, tolerance, match):
    """Linear constraints reject invalid or inapplicable tolerances."""
    with pytest.raises(ValueError, match=match):
        DiscreteLinearConstraint(
            parameters=["A", "B"],
            operator=operator,
            rhs=1.0,
            tolerance=tolerance,
        )


##### Behavior #####


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
