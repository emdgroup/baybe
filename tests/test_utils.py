"""Tests for utilities."""
import math

import numpy as np
import pandas as pd
import pytest
from pytest import param

from baybe.utils.basic import register_hooks
from baybe.utils.memory import bytes_to_human_readable
from baybe.utils.numerical import closest_element
from baybe.utils.sampling_algorithms import DiscreteSamplingMethod, sample_numerical_df

_TARGET = 1337
_CLOSEST = _TARGET + 0.1


def f_plain(arg1, arg2):
    pass


def f_annotated(arg1: str, arg2: int):
    pass


def f_annotated_one_default(arg1: str, arg2: int = 1):
    pass


def f2_plain(arg, arg3):
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


@pytest.mark.parametrize("fraction", [0.2, 0.8, 1.0, 1.2, 2.0, 2.4, 3.5])
@pytest.mark.parametrize("method", list(DiscreteSamplingMethod))
def test_discrete_sampling(fraction, method):
    """Size consistency tests for discrete sampling utility."""
    df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list("ABCD"))

    n_points = math.ceil(fraction * len(df))
    sampled = sample_numerical_df(df, n_points, method=method)

    assert (
        len(sampled) == n_points
    ), "Sampling did not return expected number of points."
    if fraction >= 1.0:
        # Ensure the entire dataframe is contained in the sampled points
        assert (
            pd.merge(df, sampled, how="left", indicator=True)["_merge"].eq("both").all()
        ), "Oversized sampling did not return all original points at least once."
    else:
        # Assure all points are unique
        assert len(sampled) == len(
            sampled.drop_duplicates()
        ), "Undersized sampling did not return unique points."


@pytest.mark.parametrize(
    ("target", "hook"),
    [
        param(
            f_annotated,
            f_annotated_one_default,
            id="hook_with_defaults",
        ),
        param(
            f_annotated_one_default,
            f_annotated,
            id="target_with_defaults",
        ),
        param(
            f_annotated,
            f_plain,
            id="hook_without_annotations",
        ),
    ],
)
def test_valid_register_hooks(target, hook):
    """Passing consistent signatures to `register_hooks` raises no error."""
    register_hooks(target, [hook])


@pytest.mark.parametrize(
    ("target", "hook"),
    [
        param(
            f_annotated,
            f2_plain,
            id="different_names",
        ),
    ],
)
def test_invalid_register_hooks(target, hook):
    """Passing inconsistent signatures to `register_hooks` raises an error."""
    with pytest.raises(TypeError):
        register_hooks(target, [hook])
