"""Tests for utilities."""

import numpy as np
import pytest
from pytest import param

from baybe.searchspace.core import SearchSpace
from baybe.utils.numerical import closest_element

_TARGET = 1337
_CLOSEST = _TARGET + 0.1


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


def test_searchspace_memory_estimate(searchspace: SearchSpace):
    """The memory estimate doesn't differ by more than 5% from the actual memory."""
    estimate = searchspace.estimate_product_space_size(searchspace.parameters)
    estimate_exp = estimate.exp_rep_bytes
    estimate_comp = estimate.comp_rep_bytes

    actual_exp = searchspace.discrete.exp_rep.memory_usage(deep=True, index=False).sum()
    actual_comp = searchspace.discrete.comp_rep.memory_usage(
        deep=True, index=False
    ).sum()

    assert 0.95 <= estimate_exp / actual_exp <= 1.05, (estimate_exp, actual_exp)
    assert 0.95 <= estimate_comp / actual_comp <= 1.05, (estimate_comp, actual_comp)
