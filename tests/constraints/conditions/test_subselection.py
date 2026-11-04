"""Tests for sub-selection conditions."""

import pytest
from pytest import param

from baybe.constraints.conditions import SubSelectionCondition
from tests.constraints.conditions.utils import apply_condition
from tests.constraints.utils import requires_fulltest

pytestmark = requires_fulltest

##### API #####


@pytest.mark.parametrize(
    ("selection", "match"),
    [
        param([], "must be >= 1", id="empty"),
        param(["a", "a"], "must be unique", id="duplicate"),
    ],
)
def test_subselection_condition_invalid_selection(selection, match):
    """Sub-selection conditions require a non-empty list of unique values."""
    with pytest.raises(ValueError, match=match):
        SubSelectionCondition(selection=selection)


##### Behavior #####


@pytest.mark.parametrize(
    ("selection", "values", "expected"),
    [
        param(["a", "b"], ["a", "b", "c", None], ["a", "b"], id="strings"),
        param([1.0, 2.0], [1, 2, 3], [1, 2], id="integers"),
        param([1.5], [1.5, 2.5, None], [1.5], id="floats"),
        param(["x"], ["a", "b"], [], id="no-match"),
        param(["a"], [None, None], [], id="all-null"),
    ],
)
def test_subselection_condition_filters_values(selection, values, expected, backend):
    """Sub-selection conditions keep only the selected values."""
    condition = SubSelectionCondition(selection=selection)
    assert apply_condition(condition, values, backend) == expected
