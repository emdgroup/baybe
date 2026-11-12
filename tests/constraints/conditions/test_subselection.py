"""Tests for sub-selection conditions."""

import narwhals.stable.v2 as nw
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
        param(["a", 1], "common type", id="mixed"),
        param([None], "must be strings, Booleans, or numbers", id="none"),
        param([float("nan")], "NaN cannot be used", id="nan"),
    ],
)
def test_subselection_condition_invalid_selection(selection, match):
    """Sub-selection conditions require unique values of a single type."""
    with pytest.raises(ValueError, match=match):
        SubSelectionCondition(selection=selection)


##### Behavior #####


@pytest.mark.parametrize(
    ("selection", "values", "expected"),
    [
        param(["a", "b"], ["a", "b", "c", None], ["a", "b"], id="strings"),
        param([1.0, 2.0], [1, 2, 3], [1, 2], id="integers"),
        param([1, 2], [1.0, 2.0, 3.0], [1.0, 2.0], id="floats"),
        param([1.5], [1.5, 2.5, None], [1.5], id="floats-nullable"),
        param([True], [True, False], [True], id="booleans"),
        param(["x"], ["a", "b"], [], id="no-match"),
    ],
)
def test_subselection_condition_filters_values(selection, values, expected, backend):
    """Sub-selection conditions keep only the selected values."""
    condition = SubSelectionCondition(selection=selection)
    assert apply_condition(condition, values, backend) == expected


@pytest.mark.parametrize(
    ("selection", "values"),
    [
        param([True], [1, 2], id="boolean-on-numeric"),
        param([1], [True, False], id="numeric-on-boolean"),
        param([1], ["a", "b"], id="numeric-on-string"),
        param([1], [None, None], id="numeric-on-all-null"),
        param(["a"], [1.0, 2.0], id="string-on-numeric"),
        param(["a"], [True, False], id="string-on-boolean"),
    ],
)
def test_subselection_condition_incompatible_dtype(selection, values, backend):
    """Sub-selection condition expressions reject incompatible dtypes."""
    dtype = nw.new_series("value", values, backend=backend).dtype
    condition = SubSelectionCondition(selection=selection)
    with pytest.raises(TypeError, match="cannot be applied to data of type"):
        condition._to_expr(nw.col("value"), dtype)
