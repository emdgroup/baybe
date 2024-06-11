"""Tests for utilities."""


import numpy as np
import pandas as pd
import pytest
from pytest import param

from baybe.utils.memory import bytes_to_human_readable
from baybe.utils.numerical import closest_element
from baybe.utils.sampling_algorithms import DiscreteSamplingMethod, sample_numerical_df

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


def test_memory_human_readable_conversion():
    """The memory conversion to human readable format is correct."""
    assert bytes_to_human_readable(1024) == (1.0, "KB")
    assert bytes_to_human_readable(1024**2) == (1.0, "MB")
    assert bytes_to_human_readable(4.3 * 1024**4) == (4.3, "TB")


@pytest.mark.parametrize("oversized", [True, False])
@pytest.mark.parametrize("method", list(DiscreteSamplingMethod))
def test_discrete_sampling(oversized, method):
    """Size consistency tests for discrete sampling utility."""
    df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list("ABCD"))

    n_points = 2 * len(df) if oversized else len(df) - 1
    sampled = sample_numerical_df(df, n_points, method=method)

    assert (
        len(sampled) == n_points
    ), "Sampling did not return expected number of points."
    if not oversized:
        assert len(sampled) == len(
            sampled.drop_duplicates()
        ), "Undersized sampling did not return unique points."
