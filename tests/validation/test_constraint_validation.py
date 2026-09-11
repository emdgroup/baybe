"""Validation tests for constraints."""

import pytest
from pytest import param

from baybe.constraints.conditions import SubSelectionCondition
from baybe.constraints.continuous import (
    ContinuousCardinalityConstraint,
    ContinuousLinearConstraint,
)
from baybe.constraints.discrete import (
    DiscreteDependenciesConstraint,
    DiscreteLinearConstraint,
    DiscretePermutationInvarianceConstraint,
    DiscreteProductConstraint,
    DiscreteRepetitionLimitConstraint,
)


@pytest.mark.parametrize(
    ("cardinalities", "error", "match"),
    [
        param(("0", 0), TypeError, "must be <class 'int'>", id="type_min"),
        param((0, "1"), TypeError, "must be <class 'int'>", id="type_max"),
        param((-1, 0), ValueError, "'min_cardinality' must be >= 0", id="loo_small"),
        param((1, 0), ValueError, "larger than the upper bound", id="wrong_order"),
        param((0, 3), ValueError, "exceed the number of parameters", id="too_large"),
        param((0, 2), ValueError, r"No constraint .* required", id="inactive"),
    ],
)
def test_invalid_cardinalities(cardinalities, error, match):
    """Providing an invalid parameter name raises an exception."""
    with pytest.raises(error, match=match):
        ContinuousCardinalityConstraint(["x", "y"], *cardinalities)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        param(
            {"n_max_repetitions": 2.0},
            TypeError,
            "must be <class 'int'>",
            id="maximum-type",
        ),
        param(
            {"n_max_repetitions": 0},
            ValueError,
            "must be >= 1",
            id="maximum-too-small",
        ),
        param(
            {"n_max_repetitions": 4},
            ValueError,
            "must be less than the number of parameters",
            id="maximum-too-large",
        ),
        param(
            {"n_max_repetitions": 3},
            ValueError,
            "meaningful constraint",
            id="maximum-only-no-op",
        ),
    ],
)
def test_invalid_max_repetitions(kwargs, error, match):
    """Invalid maximum repetition counts raise an exception."""
    with pytest.raises(error, match=match):
        DiscreteRepetitionLimitConstraint(parameters=["A", "B", "C"], **kwargs)


@pytest.mark.parametrize(
    ("coefficients", "match"),
    [
        param((1.0, 2.0), "'coefficients' list must have one", id="length-mismatch"),
        param((1.0, 0.0, 1.0), "'coefficients' must be non-zero", id="zero-coeff"),
    ],
)
def test_invalid_coefficients(coefficients, match):
    """Invalid coefficients raise a ValueError."""
    with pytest.raises(ValueError, match=match):
        DiscreteLinearConstraint(
            parameters=["A", "B", "C"],
            operator="<=",
            rhs=1.0,
            coefficients=coefficients,
        )
    with pytest.raises(ValueError, match=match):
        ContinuousLinearConstraint(
            parameters=["A", "B", "C"],
            operator="<=",
            coefficients=coefficients,
        )


def test_excluded_permutation_dependencies():
    """Excluded permutation dependencies raise a ValueError."""
    dependencies = DiscreteDependenciesConstraint(
        parameters=["Gate"],
        conditions=[SubSelectionCondition(selection=["on"])],
        affected_parameters=[["P1", "P2"]],
        exclude=True,
    )

    with pytest.raises(
        ValueError,
        match="Dependencies of a permutation invariance constraint cannot use",
    ):
        DiscretePermutationInvarianceConstraint(
            parameters=["P1", "P2"], dependencies=dependencies
        )


@pytest.mark.parametrize(
    "constraint_cls",
    [DiscreteLinearConstraint, DiscreteProductConstraint],
    ids=["linear", "product"],
)
@pytest.mark.parametrize(
    ("operator", "tolerance", "match"),
    [
        param("=", float("nan"), "cannot be 'nan'", id="nan"),
        param("=", float("inf"), "cannot be 'inf'", id="inf"),
        param("=", -float("inf"), "cannot be 'inf'", id="neg-inf"),
        param("=", 0.0, "must be > 0", id="zero"),
        param("=", -1.0, "must be > 0", id="negative"),
        param(
            ">=", 0.1, "only valid with the following operators", id="wrong-operator"
        ),
    ],
)
def test_invalid_tolerance(constraint_cls, operator, tolerance, match):
    """Invalid tolerances are rejected eagerly at construction."""
    with pytest.raises(ValueError, match=match):
        constraint_cls(
            parameters=["A", "B"],
            operator=operator,
            rhs=1.0,
            tolerance=tolerance,
        )


@pytest.mark.parametrize(
    ("args", "kwargs", "error", "match"),
    [
        param((["A", "B"],), {}, TypeError, "missing.*operator", id="missing"),
        param(
            (["A", "B"], "=", 1, 0.01, True),
            {},
            TypeError,
            "too many positional",
            id="positional-exclude",
        ),
        param(
            (["A", "B"], "="),
            {"operator": "="},
            TypeError,
            "multiple values",
            id="duplicate",
        ),
        param(
            (["A", "B"],),
            {"operator": "bad"},
            ValueError,
            "must be in",
            id="invalid-operator",
        ),
        param(
            (["A", "B"],),
            {"operator": "=", "unknown": 1},
            TypeError,
            "unexpected keyword",
            id="unknown",
        ),
    ],
)
def test_invalid_product_arguments(args, kwargs, error, match):
    """Product argument binding rejects invalid modern call shapes."""
    with pytest.raises(error, match=match):
        DiscreteProductConstraint(*args, **kwargs)
