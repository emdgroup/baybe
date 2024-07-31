"""Tests for augmentation utilities."""

import pandas as pd
import pytest
from pytest import param

from baybe.utils.augmentation import (
    df_apply_dependency_augmentation,
    df_apply_permutation_augmentation,
)


@pytest.mark.parametrize(
    ("content", "col_groups", "content_expected"),
    [
        param(  # 2 invariant cols and 1 unaffected col
            {
                "data": {
                    "A": [1, 1],
                    "B": [2, 2],
                    "C": ["x", "y"],
                },
                "index": [0, 1],
            },
            [["A"], ["B"]],
            {
                "data": {
                    "A": [1, 2, 1, 2],
                    "B": [2, 1, 2, 1],
                    "C": ["x", "x", "y", "y"],
                },
                "index": [0, 0, 1, 1],
            },
            id="2inv_1add",
        ),
        param(  # 2 invariant cols with identical row values
            {
                "data": {
                    "A": [1, 1],
                    "B": [2, 2],
                },
                "index": [0, 1],
            },
            [["A"], ["B"]],
            {
                "data": {
                    "A": [1, 2, 1, 2],
                    "B": [2, 1, 2, 1],
                },
                "index": [0, 0, 1, 1],
            },
            id="2inv+degen",
        ),
        param(  # 2 groups with identical row values but different targets
            {
                "data": {
                    "A": [1, 1],
                    "B": [2, 2],
                    "T": ["x", "y"],
                },
                "index": [0, 1],
            },
            [["A"], ["B"]],
            {
                "data": {
                    "A": [1, 2, 1, 2],
                    "B": [2, 1, 2, 1],
                    "T": ["x", "x", "y", "y"],
                },
                "index": [0, 0, 1, 1],
            },
            id="2inv+degen_target",
        ),
        param(  # 2 groups with identical row values but same targets
            {
                "data": {
                    "A": [1, 1],
                    "B": [2, 2],
                    "T": ["x", "x"],
                },
                "index": [0, 1],
            },
            [["A"], ["B"]],
            {
                "data": {
                    "A": [1, 2, 1, 2],
                    "B": [2, 1, 2, 1],
                    "T": ["x", "x", "x", "x"],
                },
                "index": [0, 0, 1, 1],
            },
            id="2inv+degen_target+degen",
        ),
        param(  # 3 invariant groups with 1 entry each
            {
                "data": {
                    "A": [1, 1],
                    "B": [2, 4],
                    "C": [3, 5],
                    "D": ["x", "y"],
                },
                "index": [0, 1],
            },
            [["A"], ["B"], ["C"]],
            {
                "data": {
                    "A": [1, 1, 2, 2, 3, 3, 1, 1, 4, 4, 5, 5],
                    "B": [2, 3, 1, 3, 1, 2, 4, 5, 1, 5, 1, 4],
                    "C": [3, 2, 3, 1, 2, 1, 5, 4, 5, 1, 4, 1],
                    "D": ["x", "x", "x", "x", "x", "x", "y", "y", "y", "y", "y", "y"],
                },
                "index": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            },
            id="3inv_1add",
        ),
        param(  # 2 groups with 2 entries each, 2 additional columns
            {
                "data": {
                    "Slot1": ["s1", "s2"],
                    "Slot2": ["s2", "s4"],
                    "Frac1": [0.1, 0.6],
                    "Frac2": [0.9, 0.4],
                    "Other1": ["A", "B"],
                    "Other2": ["C", "D"],
                },
                "index": [0, 1],
            },
            [["Slot1", "Frac1"], ["Slot2", "Frac2"]],
            {
                "data": {
                    "Slot1": ["s1", "s2", "s2", "s4"],
                    "Slot2": ["s2", "s1", "s4", "s2"],
                    "Frac1": [0.1, 0.9, 0.6, 0.4],
                    "Frac2": [0.9, 0.1, 0.4, 0.6],
                    "Other1": ["A", "A", "B", "B"],
                    "Other2": ["C", "C", "D", "D"],
                },
                "index": [0, 0, 1, 1],
            },
            id="2inv_2dependent_2add",
        ),
        param(  # 2 groups with 3 entries each, 1 additional column
            {
                "data": {
                    "Slot1": ["s1", "s2"],
                    "Slot2": ["s2", "s4"],
                    "Frac1": [0.1, 0.6],
                    "Frac2": [0.9, 0.4],
                    "Temp1": [10, 20],
                    "Temp2": [50, 60],
                    "Other": ["x", "y"],
                },
                "index": [0, 1],
            },
            [["Slot1", "Frac1", "Temp1"], ["Slot2", "Frac2", "Temp2"]],
            {
                "data": {
                    "Slot1": ["s1", "s2", "s2", "s4"],
                    "Slot2": ["s2", "s1", "s4", "s2"],
                    "Frac1": [0.1, 0.9, 0.6, 0.4],
                    "Frac2": [0.9, 0.1, 0.4, 0.6],
                    "Temp1": [10, 50, 20, 60],
                    "Temp2": [50, 10, 60, 20],
                    "Other": ["x", "x", "y", "y"],
                },
                "index": [0, 0, 1, 1],
            },
            id="2inv_4dependent2each_1add",
        ),
    ],
)
def test_df_permutation_aug(content, col_groups, content_expected):
    """Test permutation invariance data augmentation is done correctly."""
    # Create all needed dataframes
    df = pd.DataFrame(**content)
    df_augmented = df_apply_permutation_augmentation(df, col_groups)
    df_expected = pd.DataFrame(**content_expected)

    # Determine equality ignoring row order
    are_equal = df_augmented.equals(df_expected)

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
    ("content", "causing", "affected", "content_expected"),
    [
        param(  # 1 causing val, 1 col affected (with 3 values)
            {
                "data": {
                    "A": [0, 1],
                    "B": [3, 4],
                    "C": ["x", "y"],
                },
                "index": [0, 1],
            },
            ("A", [0]),
            [("B", [3, 4, 5])],
            {
                "data": {
                    "A": [0, 0, 0, 1],
                    "B": [3, 4, 5, 4],
                    "C": ["x", "x", "x", "y"],
                },
                "index": [0, 0, 0, 1],
            },
            id="1causing_1affected",
        ),
        param(  # 1 causing val, 2 cols affected (with 2 values each)
            {
                "data": {
                    "A": [0, 1],
                    "B": [3, 4],
                    "C": ["x", "y"],
                },
                "index": [0, 1],
            },
            ("A", [0]),
            [("B", [3, 4]), ("C", ["x", "y"])],
            {
                "data": {
                    "A": [0, 0, 0, 0, 1],
                    "B": [3, 3, 4, 4, 4],
                    "C": ["x", "y", "x", "y", "y"],
                },
                "index": [0, 0, 0, 0, 1],
            },
            id="1causing_2affected",
        ),
        param(  # 2 causing vals, 1 col affected (with 3 values)
            {
                "data": {
                    "A": [0, 1, 2],
                    "B": [3, 4, 3],
                    "C": ["x", "y", "z"],
                },
                "index": [0, 1, 2],
            },
            ("A", [0, 1]),
            [("B", [3, 4, 5])],
            {
                "data": {
                    "A": [0, 0, 0, 1, 1, 1, 2],
                    "B": [3, 4, 5, 3, 4, 5, 3],
                    "C": ["x", "x", "x", "y", "y", "y", "z"],
                },
                "index": [0, 0, 0, 1, 1, 1, 2],
            },
            id="2causing_1affected",
        ),
        param(  # 2 causing vals, 2 cols affected (with 2 values each)
            {
                "data": {
                    "A": [0, 1, 2],
                    "B": [3, 4, 3],
                    "C": ["x", "y", "x"],
                },
                "index": [0, 1, 2],
            },
            ("A", [0, 1]),
            [("B", [3, 4]), ("C", ["x", "y"])],
            {
                "data": {
                    "A": [0, 0, 0, 0, 1, 1, 1, 1, 2],
                    "B": [3, 3, 4, 4, 3, 3, 4, 4, 3],
                    "C": ["x", "y", "x", "y", "x", "y", "x", "y", "x"],
                },
                "index": [0, 0, 0, 0, 1, 1, 1, 1, 2],
            },
            id="2causing_2affected",
        ),
    ],
)
def test_df_dependency_aug(content, causing, affected, content_expected):
    """Test dependency data augmentation is done correctly."""
    # Create all needed dataframes
    df = pd.DataFrame(**content)
    df_augmented = df_apply_dependency_augmentation(df, causing, affected)
    df_expected = pd.DataFrame(**content_expected)

    # Determine equality ignoring row order
    are_equal = df_augmented.equals(df_expected)

    assert (
        are_equal
    ), f"\norig:\n{df}\n\naugmented:\n{df_augmented}\n\nexpected:\n{df_expected}"
