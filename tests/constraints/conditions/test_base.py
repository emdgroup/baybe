"""Tests for shared condition behavior."""

import narwhals.stable.v2 as nw
import pytest

from baybe.constraints.conditions import (
    Condition,
    SubSelectionCondition,
    ThresholdCondition,
)
from baybe.utils.basic import get_subclasses
from tests.constraints.conditions.utils import apply_condition
from tests.constraints.utils import requires_fulltest

pytestmark = requires_fulltest

_MISSING_VALUES = [None, float("nan")]
_CONDITIONS = [ThresholdCondition(2.0, "!="), SubSelectionCondition([2.0])]


def test_missing_value_contract_covers_all_condition_classes():
    """The shared missing-value cases cover every concrete condition class."""
    assert {type(condition) for condition in _CONDITIONS} == set(
        get_subclasses(Condition)
    )


@pytest.mark.parametrize(
    "condition", _CONDITIONS, ids=lambda condition: type(condition).__name__
)
def test_conditions_treat_null_and_nan_as_non_matching(condition, backend):
    """Conditions treat nulls and NaNs as non-matching during evaluation and call."""
    series = nw.new_series("value", _MISSING_VALUES, backend=backend)
    mask = nw.from_native(condition.evaluate(series.to_native()), series_only=True)

    assert mask.to_list() == [False, False]
    assert apply_condition(condition, _MISSING_VALUES, backend) == []
