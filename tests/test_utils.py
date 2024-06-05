"""Tests for utilities."""

from contextlib import nullcontext

import numpy as np
import pytest
from pytest import param

from baybe.utils.basic import register_hook
from baybe.utils.memory import bytes_to_human_readable
from baybe.utils.numerical import closest_element

_TARGET = 1337
_CLOSEST = _TARGET + 0.1


def base_func(arg1: str, arg2: int):
    pass


def func_with_default_values(arg1: str, arg2: int = 1):
    pass


def func_without_annotations(arg1, arg2):
    pass


def func_with_different_order(arg2: int, arg1: str):
    pass


def func_with_different_names(arg: int, arg3: str):
    pass


@pytest.mark.parametrize(
    "as_ndarray", [param(False, id="list"), param(True, id="array")]
)
@pytest.mark.parametrize(
    "array",
    [
        param(_CLOSEST, id="0D"),
        param([0, _CLOSEST], id="1D"),
        param([[2, 3], [0, _CLOSEST]], id="2D"),
    ],
)
def test_closest_element(as_ndarray, array):
    """The closest element can be found irrespective of the input type."""
    if as_ndarray:
        array = np.asarray(array)
    actual = closest_element(array, _TARGET)
    assert actual == _CLOSEST, (actual, _CLOSEST)


def test_memory_human_readable_conversion():
    """The memory conversion to human readable format is correct."""
    assert bytes_to_human_readable(1024) == (1.0, "KB")
    assert bytes_to_human_readable(1024**2) == (1.0, "MB")
    assert bytes_to_human_readable(4.3 * 1024**4) == (4.3, "TB")


@pytest.mark.parametrize(
    ("target, hook, error"),
    [
        param(base_func, func_with_default_values, None, id="hook_with_default_values"),
        param(
            func_with_default_values, base_func, None, id="target_with_default_values"
        ),
        param(base_func, func_without_annotations, None, id="hook_has_no_annotations"),
        param(
            base_func,
            func_with_different_order,
            TypeError,
            id="hook_has_different_signature_order",
        ),
        param(
            func_with_different_order,
            base_func,
            TypeError,
            id="target_has_different_signature_order",
        ),
        param(
            base_func,
            func_with_different_names,
            TypeError,
            id="hook_has_different_names",
        ),
        param(
            func_with_different_names,
            base_func,
            TypeError,
            id="target_has_different_names",
        ),
    ],
)
def test_register_hook(target, hook, error):
    """Test the behavior of the register_hook function with different scenarios."""
    with pytest.raises(error) if error is not None else nullcontext():
        register_hook(target, hook)
