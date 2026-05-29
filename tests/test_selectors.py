"""Tests for parameter selectors."""

import pytest
from pytest import param

from baybe.parameters import (
    CategoricalParameter,
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
    TaskParameter,
)
from baybe.parameters.selectors import NameSelector, TypeSelector, to_parameter_selector

_PARAMS = (
    CategoricalParameter("cat", ["a", "b"]),
    TaskParameter("task", ["t1", "t2"]),
    NumericalDiscreteParameter("num", [1, 2]),
    NumericalContinuousParameter("cont", (0, 1)),
)


@pytest.mark.parametrize(
    ("input_val", "expected"),
    [
        param(
            NameSelector(["cat", "num"]),
            (True, False, True, False),
            id="name_selector",
        ),
        param(
            TypeSelector((TaskParameter,)),
            (False, True, False, False),
            id="type_selector",
        ),
        param(
            TypeSelector((TaskParameter,), exclude=True),
            (True, False, True, True),
            id="type_selector_exclude",
        ),
        param(
            NameSelector(["c.*"], regex=True),
            (True, False, False, True),
            id="name_selector_regex",
        ),
        param(
            lambda p: p.name.startswith("c"),
            (True, False, False, True),
            id="custom_callable",
        ),
        param("task", (False, True, False, False), id="bare_string"),
        param(TaskParameter, (False, True, False, False), id="bare_type"),
        param(["cat", "cont"], (True, False, False, True), id="string_list"),
        param(
            [CategoricalParameter, NumericalContinuousParameter],
            (True, True, False, True),
            id="type_list",
        ),
        # --- Error cases ---
        param(["cat", TaskParameter], TypeError, id="mixed_collection"),
        param(42, TypeError, id="invalid_type"),
        param([int, str], TypeError, id="non_parameter_types"),
    ],
)
def test_to_parameter_selector(input_val, expected):
    """Converted selectors select the right parameters."""
    if expected is TypeError:
        with pytest.raises(TypeError):
            to_parameter_selector(input_val)
    else:
        selector = to_parameter_selector(input_val)
        result = tuple(selector(p) for p in _PARAMS)
        assert result == expected
