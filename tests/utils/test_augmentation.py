"""Tests for augmentation utilities."""


import pandas as pd
import pytest
from pytest import param

from baybe.utils.augmentation import (
    df_apply_dependency_augmentation,
    df_apply_permutation_augmentation,
)


@pytest.mark.parametrize(
    ("data", "col_groups", "data_expected"),
    [
        param(  # 2 invariant cols and 1 unaffected col
            {
                "A": [1, 1],
                "B": [2, 2],
                "C": ["x", "y"],
            },
            [["A"], ["B"]],
            {
                "A": [1, 2, 1, 2],
                "B": [2, 1, 2, 1],
                "C": ["x", "x", "y", "y"],
            },
            id="2inv_1add",
        ),
        param(  # 2 invariant cols with identical values
            {
                "A": [1, 1],
                "B": [2, 2],
            },
            [["A"], ["B"]],
            {
                "A": [1, 1, 2],
                "B": [2, 2, 1],
            },
            id="2inv+degen",
        ),
        param(  # 2 invariant cols with identical values but different targets
            {
                "A": [1, 1],
                "B": [2, 2],
                "T": ["x", "y"],
            },
            [["A"], ["B"]],
            {
                "A": [1, 1, 2, 2],
                "B": [2, 2, 1, 1],
                "T": ["x", "y", "x", "y"],
            },
            id="2inv+degen_target",
        ),
        param(  # 2 invariant cols with identical values but different targets
            {
                "A": [1, 1],
                "B": [2, 2],
                "T": ["x", "x"],
            },
            [["A"], ["B"]],
            {
                "A": [1, 2],
                "B": [2, 1],
                "T": ["x", "x"],
            },
            id="2inv+degen_target+degen",
        ),
        param(  # 3 invariant cols
            {
                "A": [1, 1],
                "B": [2, 4],
                "C": [3, 5],
                "D": ["x", "y"],
            },
            [["A"], ["B"], ["C"]],
            {
                "A": [1, 1, 2, 2, 3, 3, 1, 1, 4, 4, 5, 5],
                "B": [2, 3, 1, 3, 2, 1, 4, 5, 1, 5, 1, 4],
                "C": [3, 2, 3, 1, 1, 2, 5, 4, 5, 1, 4, 1],
                "D": ["x", "x", "x", "x", "x", "x", "y", "y", "y", "y", "y", "y"],
            },
            id="3inv_1add",
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
            [["Slot1", "Frac1"], ["Slot2", "Frac2"]],
            {
                "Slot1": ["s1", "s2", "s2", "s4"],
                "Slot2": ["s2", "s4", "s1", "s2"],
                "Frac1": [0.1, 0.6, 0.9, 0.4],
                "Frac2": [0.9, 0.4, 0.1, 0.6],
                "Other1": ["A", "B", "A", "B"],
                "Other2": ["C", "D", "C", "D"],
            },
            id="2inv_2dependent_2add",
        ),
        param(  # 2 invariant cols, 2 dependent ones, 2 additional ones
            {
                "Slot1": ["s1", "s2"],
                "Slot2": ["s2", "s4"],
                "Frac1": [0.1, 0.6],
                "Frac2": [0.9, 0.4],
                "Temp1": [10, 20],
                "Temp2": [50, 60],
                "Other": ["x", "y"],
            },
            [["Slot1", "Frac1", "Temp1"], ["Slot2", "Frac2", "Temp2"]],
            {
                "Slot1": ["s1", "s2", "s2", "s4"],
                "Slot2": ["s2", "s4", "s1", "s2"],
                "Frac1": [0.1, 0.6, 0.9, 0.4],
                "Frac2": [0.9, 0.4, 0.1, 0.6],
                "Temp1": [10, 20, 50, 60],
                "Temp2": [50, 60, 10, 20],
                "Other": ["x", "y", "x", "y"],
            },
            id="2inv_4dependent2each_1add",
        ),
    ],
)
def test_df_permutation_aug(data, col_groups, data_expected):
    """Test permutation invariance data augmentation is done correctly."""
    # Create all needed dataframes
    df = pd.DataFrame(data)
    df_augmented = df_apply_permutation_augmentation(df, col_groups)
    df_expected = pd.DataFrame(data_expected)

    # Determine equality ignoring row order
    are_equal = (
        pd.merge(left=df_augmented, right=df_expected, how="outer", indicator=True)[
            "_merge"
        ]
        .eq("both")
        .all()
    )

    assert (
        are_equal
    ), f"\norig:\n{df}\n\naugmented:\n{df_augmented}\n\nexpected:\n{df_expected}"


@pytest.mark.parametrize(
    ("col_groups", "msg"),
    [
        param([], "at least two column sequences", id="no_groups"),
        param([["A"]], "at least two column sequences", id="just_one_group"),
        param([["A"], ["B", "C"]], "the amount of columns in", id="different_lengths"),
        param([[], []], "each column group has", id="empty_group"),
    ],
)
def test_df_permutation_aug_invalid(col_groups, msg):
    """Test correct errors for invalid permutation attempts."""
    df = pd.DataFrame({"A": [1, 1], "B": [2, 2], "C": ["x", "y"]})
    with pytest.raises(ValueError, match=msg):
        df_apply_permutation_augmentation(df, col_groups)


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
def test_df_dependency_aug(data, causing, affected, data_expected):
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

    assert (
        are_equal
    ), f"\norig:\n{df}\n\naugmented:\n{df_augmented}\n\nexpected:\n{df_expected}"
