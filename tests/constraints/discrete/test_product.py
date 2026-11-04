"""Tests for the discrete product constraint."""

import pytest
from pytest import param

from baybe.constraints.discrete import DiscreteProductConstraint
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
def test_product_constraint_invalid_parameters(parameters, error, match):
    """Product constraints require a non-empty list of unique parameter names."""
    with pytest.raises(error, match=match):
        DiscreteProductConstraint(
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
def test_product_constraint_invalid_operator(operator):
    """Product constraints reject unknown comparison operators."""
    with pytest.raises(ValueError, match="'operator' must be in"):
        DiscreteProductConstraint(parameters=["A", "B"], operator=operator, rhs=1.0)


@pytest.mark.parametrize(
    ("rhs", "match"),
    [
        param(float("nan"), "cannot be 'nan'", id="nan"),
        param(float("inf"), "cannot be 'inf'", id="inf"),
        param(-float("inf"), "cannot be 'inf'", id="neg-inf"),
    ],
)
def test_product_constraint_invalid_rhs(rhs, match):
    """Product constraints reject non-finite right-hand sides."""
    with pytest.raises(ValueError, match=match):
        DiscreteProductConstraint(parameters=["A", "B"], operator="<=", rhs=rhs)


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
def test_product_constraint_invalid_tolerance(operator, tolerance, match):
    """Product constraints reject invalid or inapplicable tolerances."""
    with pytest.raises(ValueError, match=match):
        DiscreteProductConstraint(
            parameters=["A", "B"],
            operator=operator,
            rhs=1.0,
            tolerance=tolerance,
        )


##### Behavior #####


_PRODUCT_DATA = [
    {"amount_1": 0.0, "amount_2": 0.0, "solvent": "water"},
    {"amount_1": 1.0, "amount_2": 0.0, "solvent": "ethanol"},
    {"amount_1": 1.0, "amount_2": 1.0, "solvent": "methanol"},
    {"amount_1": 2.0, "amount_2": 1.0, "solvent": "acetone"},
    {"amount_1": 2.0, "amount_2": 2.0, "solvent": "water"},
    {"amount_1": 5.0, "amount_2": 2.0, "solvent": "hexane"},
]


@pytest.mark.parametrize(
    ("operator", "rhs", "tolerance", "exclude", "expected_rows"),
    [
        param(
            "<",
            2.0,
            None,
            False,
            [
                {"amount_1": 0.0, "amount_2": 0.0, "solvent": "water"},
                {"amount_1": 1.0, "amount_2": 0.0, "solvent": "ethanol"},
                {"amount_1": 1.0, "amount_2": 1.0, "solvent": "methanol"},
            ],
            id="operator-lt",
        ),
        param(
            "<=",
            2.0,
            None,
            False,
            [
                {"amount_1": 0.0, "amount_2": 0.0, "solvent": "water"},
                {"amount_1": 1.0, "amount_2": 0.0, "solvent": "ethanol"},
                {"amount_1": 1.0, "amount_2": 1.0, "solvent": "methanol"},
                {"amount_1": 2.0, "amount_2": 1.0, "solvent": "acetone"},
            ],
            id="operator-le",
        ),
        param(
            "=",
            2.0,
            None,
            False,
            [{"amount_1": 2.0, "amount_2": 1.0, "solvent": "acetone"}],
            id="operator-eq",
        ),
        param(
            "==",
            2.0,
            None,
            False,
            [{"amount_1": 2.0, "amount_2": 1.0, "solvent": "acetone"}],
            id="operator-eq-alias",
        ),
        param(
            "!=",
            2.0,
            None,
            False,
            [
                {"amount_1": 0.0, "amount_2": 0.0, "solvent": "water"},
                {"amount_1": 1.0, "amount_2": 0.0, "solvent": "ethanol"},
                {"amount_1": 1.0, "amount_2": 1.0, "solvent": "methanol"},
                {"amount_1": 2.0, "amount_2": 2.0, "solvent": "water"},
                {"amount_1": 5.0, "amount_2": 2.0, "solvent": "hexane"},
            ],
            id="operator-ne",
        ),
        param(
            ">",
            2.0,
            None,
            False,
            [
                {"amount_1": 2.0, "amount_2": 2.0, "solvent": "water"},
                {"amount_1": 5.0, "amount_2": 2.0, "solvent": "hexane"},
            ],
            id="operator-gt",
        ),
        param(
            ">=",
            2.0,
            None,
            False,
            [
                {"amount_1": 2.0, "amount_2": 1.0, "solvent": "acetone"},
                {"amount_1": 2.0, "amount_2": 2.0, "solvent": "water"},
                {"amount_1": 5.0, "amount_2": 2.0, "solvent": "hexane"},
            ],
            id="operator-ge",
        ),
        param(
            "=",
            1.5,
            0.5,
            False,
            [
                {"amount_1": 1.0, "amount_2": 1.0, "solvent": "methanol"},
                {"amount_1": 2.0, "amount_2": 1.0, "solvent": "acetone"},
            ],
            id="tolerance",
        ),
        param(
            "<=",
            2.0,
            None,
            True,
            [
                {"amount_1": 2.0, "amount_2": 2.0, "solvent": "water"},
                {"amount_1": 5.0, "amount_2": 2.0, "solvent": "hexane"},
            ],
            id="exclude",
        ),
    ],
)
def test_product_constraint(operator, rhs, tolerance, exclude, expected_rows, backend):
    """Cover all product-constraint comparison regimes."""
    constraint = DiscreteProductConstraint(
        parameters=["amount_1", "amount_2"],
        operator=operator,
        rhs=rhs,
        tolerance=tolerance,
        exclude=exclude,
    )
    assert_filter(constraint, _PRODUCT_DATA, expected_rows, backend)


def test_product_constraint_partial_rejected(backend):
    """Product constraints defer until all parameters are available."""
    constraint = DiscreteProductConstraint(
        parameters=["amount_1", "amount_2"],
        operator="<=",
        rhs=2.0,
    )
    assert_partial_rejected(constraint, [{"amount_1": 1.0}], backend)
