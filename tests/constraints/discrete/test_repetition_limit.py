"""Tests for the discrete repetition limit constraint."""

import narwhals.stable.v2 as nw
import pytest
from pytest import param

from baybe.constraints.discrete import DiscreteRepetitionLimitConstraint
from tests.constraints.discrete.utils import (
    assert_filter,
    assert_partial_rejected,
    requires_fulltest,
)

pytestmark = requires_fulltest

##### API #####


@pytest.mark.parametrize(
    ("n_max_repetitions", "error", "match"),
    [
        param(2.0, TypeError, "must be <class 'int'>", id="type"),
        param(0, ValueError, "must be >= 1", id="too-small"),
        param(4, ValueError, "must be less than the number", id="too-large"),
        param(3, ValueError, "meaningful constraint", id="no-op"),
    ],
)
def test_repetition_limit_constraint_invalid_max_repetitions(
    n_max_repetitions, error, match
):
    """Repetition limits reject invalid maximum repetition counts."""
    with pytest.raises(error, match=match):
        DiscreteRepetitionLimitConstraint(
            parameters=["A", "B", "C"],
            n_max_repetitions=n_max_repetitions,
        )


##### Behavior #####


_REPETITION_DATA = [
    {"a": 1.0, "b": 2.0, "c": 3.0, "group": "a"},
    {"a": 1.0, "b": 1.0, "c": 2.0, "group": "b"},
    {"a": 1.0, "b": 2.0, "c": 1.0, "group": "c"},
    {"a": 1.0, "b": 1.0, "c": 1.0, "group": "d"},
    {"a": 1.0, "b": 2.0, "c": 2.0, "group": "e"},
]


_REPETITION_PARTIAL_DATA = (
    nw.from_dicts(_REPETITION_DATA, backend=nw.Implementation.POLARS)
    .select("a", "b", "group")
    .rows(named=True)
)


@pytest.mark.parametrize(
    ("n_max_repetitions", "exclude", "expected_rows"),
    [
        param(
            1,
            False,
            [{"a": 1.0, "b": 2.0, "c": 3.0, "group": "a"}],
            id="max-one-keep",
        ),
        param(
            1,
            True,
            [
                {"a": 1.0, "b": 1.0, "c": 2.0, "group": "b"},
                {"a": 1.0, "b": 2.0, "c": 1.0, "group": "c"},
                {"a": 1.0, "b": 1.0, "c": 1.0, "group": "d"},
                {"a": 1.0, "b": 2.0, "c": 2.0, "group": "e"},
            ],
            id="max-one-exclude",
        ),
        param(
            2,
            False,
            [
                {"a": 1.0, "b": 2.0, "c": 3.0, "group": "a"},
                {"a": 1.0, "b": 1.0, "c": 2.0, "group": "b"},
                {"a": 1.0, "b": 2.0, "c": 1.0, "group": "c"},
                {"a": 1.0, "b": 2.0, "c": 2.0, "group": "e"},
            ],
            id="max-two-keep",
        ),
        param(
            2,
            True,
            [{"a": 1.0, "b": 1.0, "c": 1.0, "group": "d"}],
            id="max-two-exclude",
        ),
    ],
)
def test_repetition_limit_constraint(
    n_max_repetitions, exclude, expected_rows, backend
):
    """Repetition limits cover the maximum-repeat and exclusion regimes."""
    constraint = DiscreteRepetitionLimitConstraint(
        parameters=["a", "b", "c"],
        n_max_repetitions=n_max_repetitions,
        exclude=exclude,
    )
    assert_filter(constraint, _REPETITION_DATA, expected_rows, backend)


def test_repetition_limit_constraint_dtype_mismatch(backend):
    """Values of incompatible dtypes do not count as repetitions."""
    data = [{"a": 1.0, "b": "1", "c": 2.0, "group": "a"}]
    constraint = DiscreteRepetitionLimitConstraint(
        parameters=["a", "b", "c"],
        n_max_repetitions=1,
    )
    assert_filter(constraint, data, data, backend)


@pytest.mark.parametrize(
    ("n_max_repetitions", "exclude", "expected_rows"),
    [
        param(
            1,
            False,
            [
                {"a": 1.0, "b": 2.0, "group": "a"},
                {"a": 1.0, "b": 2.0, "group": "c"},
                {"a": 1.0, "b": 2.0, "group": "e"},
            ],
            id="max-one-keep",
        ),
        param(
            2,
            True,
            [
                {"a": 1.0, "b": 1.0, "group": "b"},
                {"a": 1.0, "b": 1.0, "group": "d"},
            ],
            id="max-two-exclude",
        ),
    ],
)
def test_repetition_limit_constraint_partial_accepted(
    n_max_repetitions, exclude, expected_rows, backend
):
    """Accepted partial repetition filtering uses the available columns."""
    constraint = DiscreteRepetitionLimitConstraint(
        parameters=["a", "b", "c"],
        n_max_repetitions=n_max_repetitions,
        exclude=exclude,
    )
    assert_filter(constraint, _REPETITION_PARTIAL_DATA, expected_rows, backend)


@pytest.mark.parametrize(
    ("n_max_repetitions", "exclude", "data"),
    [
        param(1, False, [{"a": 1.0}], id="too-few-columns"),
        param(1, True, _REPETITION_PARTIAL_DATA, id="exclude-needs-all"),
    ],
)
def test_repetition_limit_constraint_partial_rejected(
    n_max_repetitions, exclude, data, backend
):
    """Repetition limits defer when too few constrained columns are available."""
    constraint = DiscreteRepetitionLimitConstraint(
        parameters=["a", "b", "c"],
        n_max_repetitions=n_max_repetitions,
        exclude=exclude,
    )
    assert_partial_rejected(constraint, data, backend)
