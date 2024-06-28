"""Tests for utilities."""
import math

import numpy as np
import pandas as pd
import pytest
from pytest import param

from baybe.utils.augmentation import (
    df_apply_dependency_augmentation,
    df_apply_permutation_augmentation,
)
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


@pytest.mark.parametrize(
    ("data", "columns", "dependents", "data_expected"),
    [
        param(  # 2 invariant cols and 1 unaffected col
            {
                "A": [1, 1],
                "B": [2, 2],
                "C": ["x", "y"],
            },
            ["A", "B"],
            None,
            {
                "A": [1, 2, 1, 2],
                "B": [2, 1, 2, 1],
                "C": ["x", "x", "y", "y"],
            },
            id="2inv+1add",
        ),
        param(  # 2 invariant cols with identical values
            {"A": [1, 1], "B": [2, 2]},
            ["A", "B"],
            None,
            {
                "A": [1, 2],
                "B": [2, 1],
            },
            id="2inv_degen",
        ),
        param(  # 2 invariant cols with identical values but different targets
            {"A": [1, 1], "B": [2, 2], "T": ["x", "y"]},
            ["A", "B"],
            None,
            {
                "A": [1, 1, 2, 2],
                "B": [2, 2, 1, 1],
                "T": ["x", "y", "x", "y"],
            },
            id="2inv_degen+target_unique",
        ),
        param(  # 2 invariant cols with identical values but different targets
            {"A": [1, 1], "B": [2, 2], "T": ["x", "x"]},
            ["A", "B"],
            None,
            {
                "A": [1, 2],
                "B": [2, 1],
                "T": ["x", "x"],
            },
            id="2inv_degen+target_degen",
        ),
        param(  # 3 invariant cols
            {"A": [1, 1], "B": [2, 4], "C": [3, 5]},
            ["A", "B", "C"],
            None,
            {
                "A": [1, 1, 2, 2, 3, 3, 1, 1, 4, 4, 5, 5],
                "B": [2, 3, 1, 3, 2, 1, 4, 5, 1, 5, 1, 4],
                "C": [3, 2, 3, 1, 1, 2, 5, 4, 5, 1, 4, 1],
            },
            id="3inv",
        ),
        param(  # 3 invariant cols
            {"A": [1, 1], "B": [2, 4], "C": [3, 5], "D": ["x", "y"]},
            ["A", "B", "C"],
            None,
            {
                "A": [1, 1, 2, 2, 3, 3, 1, 1, 4, 4, 5, 5],
                "B": [2, 3, 1, 3, 2, 1, 4, 5, 1, 5, 1, 4],
                "C": [3, 2, 3, 1, 1, 2, 5, 4, 5, 1, 4, 1],
                "D": ["x", "x", "x", "x", "x", "x", "y", "y", "y", "y", "y", "y"],
            },
            id="3inv+1add",
        ),
        param(  # 2 invariant cols, 2 dependent ones, 2 additional ones
            {
                "Slot1": ["s1", "s2"],
                "Slot2": ["s2", "s4"],
                "Frac1": [0.1, 0.6],
                "Frac2": [0.9, 0.4],
                "Other1": ["A", "B"],
                "Other2": ["C", "D"],
            },
            ["Slot1", "Slot2"],
            ["Frac1", "Frac2"],
            {
                "Slot1": ["s1", "s2", "s2", "s4"],
                "Slot2": ["s2", "s4", "s1", "s2"],
                "Frac1": [0.1, 0.6, 0.9, 0.4],
                "Frac2": [0.9, 0.4, 0.1, 0.6],
                "Other1": ["A", "B", "A", "B"],
                "Other2": ["C", "D", "C", "D"],
            },
            id="2inv_degen+2dependent+2add",
        ),
    ],
)
def test_df_invariance_augmentation(data, columns, dependents, data_expected):
    """Test invariance data augmentation is done correctly."""
    # Create all needed dataframes
    df = pd.DataFrame(data)
    df_augmented = df_apply_permutation_augmentation(df, columns, dependents)
    df_expected = pd.DataFrame(data_expected)

    # Determine equality ignoring row order
    are_equal = (
        pd.merge(left=df_augmented, right=df_expected, how="outer", indicator=True)[
            "_merge"
        ]
        .eq("both")
        .all()
    )

    assert are_equal, (df, df_augmented, df_expected)


@pytest.mark.parametrize(
    ("data", "causing", "affected", "data_expected"),
    [
        param(  # 1 causing val, 1 col affected (with 3 values)
            {
                "A": [0, 1],
                "B": [3, 4],
                "C": ["x", "y"],
            },
            ("A", [0]),
            [("B", [3, 4, 5])],
            {
                "A": [0, 1, 0, 0],
                "B": [3, 4, 4, 5],
                "C": ["x", "y", "x", "x"],
            },
            id="1causing_1affected",
        ),
        param(  # 1 causing val, 2 cols affected (with 2 values each)
            {
                "A": [0, 1],
                "B": [3, 4],
                "C": ["x", "y"],
            },
            ("A", [0]),
            [("B", [3, 4]), ("C", ["x", "y"])],
            {
                "A": [0, 1, 0, 0, 0],
                "B": [3, 4, 4, 3, 4],
                "C": ["x", "y", "x", "y", "y"],
            },
            id="1causing_2affected",
        ),
        param(  # 2 causing vals, 1 col affected (with 3 values)
            {
                "A": [0, 1, 2],
                "B": [3, 4, 3],
                "C": ["x", "y", "z"],
            },
            ("A", [0, 1]),
            [("B", [3, 4, 5])],
            {
                "A": [0, 1, 2, 0, 0, 1, 1],
                "B": [3, 4, 3, 4, 5, 3, 5],
                "C": ["x", "y", "z", "x", "x", "y", "y"],
            },
            id="2causing_1affected",
        ),
        param(  # 2 causing vals, 2 cols affected (with 2 values each)
            {
                "A": [0, 1, 2],
                "B": [3, 4, 3],
                "C": ["x", "y", "x"],
            },
            ("A", [0, 1]),
            [("B", [3, 4]), ("C", ["x", "y"])],
            {
                "A": [0, 1, 2, 0, 0, 0, 1, 1, 1],
                "B": [3, 4, 3, 4, 3, 4, 3, 3, 4],
                "C": ["x", "y", "x", "x", "y", "y", "y", "x", "x"],
            },
            id="2causing_2affected",
        ),
    ],
)
def test_df_dependency_augmentation(data, causing, affected, data_expected):
    """Test dependency data augmentation is done correctly."""
    # Create all needed dataframes
    df = pd.DataFrame(data)
    df_augmented = df_apply_dependency_augmentation(df, causing, affected)
    df_expected = pd.DataFrame(data_expected)

    # Determine equality ignoring row order
    are_equal = (
        pd.merge(left=df_augmented, right=df_expected, how="outer", indicator=True)[
            "_merge"
        ]
        .eq("both")
        .all()
    )

    assert are_equal, (df, df_augmented, df_expected)
