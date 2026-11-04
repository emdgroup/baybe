"""Tests for the discrete cardinality constraint."""

import narwhals.stable.v2 as nw
import pytest
from pytest import param

from baybe.constraints.discrete import DiscreteCardinalityConstraint
from tests.constraints.discrete.utils import (
    assert_filter,
    assert_partial_rejected,
)
from tests.constraints.utils import requires_fulltest

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
def test_cardinality_constraint_invalid_parameters(parameters, error, match):
    """Cardinality constraints require a non-empty list of unique parameter names."""
    with pytest.raises(error, match=match):
        DiscreteCardinalityConstraint(
            parameters=parameters,
        )


@pytest.mark.parametrize(
    ("min_cardinality", "max_cardinality", "error", "match"),
    [
        param("0", 0, TypeError, "must be <class 'int'>", id="type-min"),
        param(0, "1", TypeError, "must be <class 'int'>", id="type-max"),
        param(-1, 0, ValueError, "'min_cardinality' must be >= 0", id="too-small"),
        param(1, 0, ValueError, "larger than the upper bound", id="wrong-order"),
        param(0, 3, ValueError, "exceed the number of parameters", id="too-large"),
        param(0, 2, ValueError, r"No constraint .* required", id="inactive"),
    ],
)
def test_cardinality_constraint_invalid_bounds(
    min_cardinality, max_cardinality, error, match
):
    """Cardinality constraints reject invalid bounds."""
    with pytest.raises(error, match=match):
        DiscreteCardinalityConstraint(
            parameters=["amount_1", "amount_2"],
            min_cardinality=min_cardinality,
            max_cardinality=max_cardinality,
        )


##### Behavior #####


_CARDINALITY_DATA = [
    {"amount_1": 0.0, "amount_2": 0.0, "amount_3": 0.0, "group": "a"},
    {"amount_1": 1.0, "amount_2": 0.0, "amount_3": 0.0, "group": "b"},
    {"amount_1": 1.0, "amount_2": 1.0, "amount_3": 0.0, "group": "c"},
    {"amount_1": 1.0, "amount_2": 1.0, "amount_3": 1.0, "group": "d"},
]


_CARDINALITY_PARTIAL_DATA = (
    nw.from_dicts(_CARDINALITY_DATA, backend=nw.Implementation.POLARS)
    .select("amount_1", "amount_2", "group")
    .rows(named=True)
)


@pytest.mark.parametrize(
    ("min_cardinality", "max_cardinality", "exclude", "expected_rows"),
    [
        param(
            1,
            2,
            False,
            [
                {"amount_1": 1.0, "amount_2": 0.0, "amount_3": 0.0, "group": "b"},
                {"amount_1": 1.0, "amount_2": 1.0, "amount_3": 0.0, "group": "c"},
            ],
            id="bounds",
        ),
        param(
            1,
            2,
            True,
            [
                {"amount_1": 0.0, "amount_2": 0.0, "amount_3": 0.0, "group": "a"},
                {"amount_1": 1.0, "amount_2": 1.0, "amount_3": 1.0, "group": "d"},
            ],
            id="bounds-exclude",
        ),
        param(
            0,
            1,
            False,
            [
                {"amount_1": 0.0, "amount_2": 0.0, "amount_3": 0.0, "group": "a"},
                {"amount_1": 1.0, "amount_2": 0.0, "amount_3": 0.0, "group": "b"},
            ],
            id="upper-only",
        ),
        param(
            2,
            2,
            False,
            [
                {"amount_1": 1.0, "amount_2": 1.0, "amount_3": 0.0, "group": "c"},
            ],
            id="exact",
        ),
        param(
            1,
            3,
            False,
            [
                {"amount_1": 1.0, "amount_2": 0.0, "amount_3": 0.0, "group": "b"},
                {"amount_1": 1.0, "amount_2": 1.0, "amount_3": 0.0, "group": "c"},
                {"amount_1": 1.0, "amount_2": 1.0, "amount_3": 1.0, "group": "d"},
            ],
            id="lower-only",
        ),
        param(
            1,
            3,
            True,
            [
                {"amount_1": 0.0, "amount_2": 0.0, "amount_3": 0.0, "group": "a"},
            ],
            id="lower-only-exclude",
        ),
    ],
)
def test_cardinality_constraint(
    min_cardinality, max_cardinality, exclude, expected_rows, backend
):
    """Cardinality constraints cover bound and exclusion regimes."""
    constraint = DiscreteCardinalityConstraint(
        parameters=["amount_1", "amount_2", "amount_3"],
        min_cardinality=min_cardinality,
        max_cardinality=max_cardinality,
        exclude=exclude,
    )
    assert_filter(constraint, _CARDINALITY_DATA, expected_rows, backend)


def test_cardinality_constraint_partial_accepted(backend):
    """Max-only cardinality filtering is valid on a non-empty subset."""
    constraint = DiscreteCardinalityConstraint(
        parameters=["amount_1", "amount_2", "amount_3"],
        max_cardinality=1,
    )
    assert_filter(
        constraint,
        _CARDINALITY_PARTIAL_DATA,
        [
            {"amount_1": 0.0, "amount_2": 0.0, "group": "a"},
            {"amount_1": 1.0, "amount_2": 0.0, "group": "b"},
        ],
        backend,
    )


def test_cardinality_constraint_partial_rejected(backend):
    """Excluded cardinality constraints require all parameters."""
    constraint = DiscreteCardinalityConstraint(
        parameters=["amount_1", "amount_2", "amount_3"],
        min_cardinality=2,
        exclude=True,
    )
    assert_partial_rejected(constraint, _CARDINALITY_PARTIAL_DATA, backend)
