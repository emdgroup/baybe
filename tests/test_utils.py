"""Tests for utilities."""

import numpy as np
import pytest
from pytest import param

from baybe.utils.numerical import closest_element


@pytest.mark.parametrize(
    "as_ndarray", [param(False, id="list"), param(True, id="array")]
)
@pytest.mark.parametrize(
    ("array", "target", "expected"),
    [
        param(0, 0.1, 0, id="0D"),
        param([0, 1], 0.1, 0, id="1D"),
        param([[2, 3], [0, 1]], 0.1, 0, id="2D"),
    ],
)
def test_closest_element(as_ndarray, array, target, expected):
    """The closest element can be found irrespective of the input type."""
    if as_ndarray:
        array = np.asarray(array)
    actual = closest_element(array, target)
    assert actual == expected, (actual, expected)
