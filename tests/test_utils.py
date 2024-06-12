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


def f_plain(arg1, arg2):
    pass


def f_reduced_plain(arg1):
    pass


def f_annotated(arg1: str, arg2: int):
    pass


def f_annotated_one_default(arg1: str, arg2: int = 1):
    pass


def f_reversed_annotated(arg2: int, arg1: str):
    pass


def f2_annotated(arg: int, arg3: str):
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
        param(
            f_annotated,
            f_annotated_one_default,
            None,
            id="hook_with_defaults",
        ),
        param(
            f_annotated_one_default,
            f_annotated,
            None,
            id="target_with_defaults",
        ),
        param(
            f_annotated,
            f_plain,
            None,
            id="hook_without_annotations",
        ),
        param(
            f_annotated,
            f_reversed_annotated,
            TypeError,
            id="different_order",
        ),
        param(
            f_annotated,
            f2_annotated,
            TypeError,
            id="different_names",
        ),
        param(
            f_annotated,
            f_reduced_plain,
            TypeError,
            id="hook_missing_arguments",
        ),
    ],
)
def test_register_hook(target, hook, error):
    """Passing in-/consistent signatures to `register_hook` raises an/no error."""
    with pytest.raises(error) if error is not None else nullcontext():
        register_hook(target, hook)
