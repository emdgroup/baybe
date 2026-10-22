"""Tests for dataframe utilities."""

from contextlib import nullcontext

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pytest import param

from baybe.exceptions import InputDataTypeWarning
from baybe.targets import BinaryTarget, NumericalTarget
from baybe.utils.dataframe import (
    _fuzzy_anti_join,
    add_noise_to_perturb_degenerate_rows,
    add_parameter_noise,
    handle_missing_values,
    normalize_input_dtypes,
)


def test_degenerate_rows():
    """Test noise-based deduplication of degenerate rows."""
    # Create random dataframe
    df = pd.DataFrame(np.random.randint(0, 100, size=(5, 3))).astype(float)

    # Manually create some degenerate rows
    df.loc[1] = df.loc[0]  # Make row 1 identical to row 0
    df.loc[3] = df.loc[2]  # Make row 3 identical to row 2
    df.iloc[:, -1] = 50.0  # Make last column constant to test the edge case

    # Add noise
    add_noise_to_perturb_degenerate_rows(df)

    # Assert that the utility fixed the degenerate rows
    assert not df.duplicated().any(), "Degenerate rows were not fixed by the utility."


def test_degenerate_rows_invalid_input():
    """Test that the utility correctly handles invalid input."""
    # Create random dataframe
    df = pd.DataFrame(np.random.randint(0, 100, size=(5, 3))).astype(float)

    # Manually create some degenerate rows
    df.loc[1] = df.loc[0]  # Make row 1 identical to row 0
    df.loc[3] = df.loc[2]  # Make row 3 identical to row 2

    # Insert invalid data types
    df = df.astype(object)  # to avoid pandas column dtype warnings
    df["invalid"] = "A"
    df.iloc[1, 0] = "B"

    # Add noise
    with pytest.raises(TypeError):
        add_noise_to_perturb_degenerate_rows(df)


@pytest.mark.parametrize(
    "parameter_names",
    [
        param(
            ["Categorical_1", "Num_disc_1", "Some_Setting"],
            id="fuzzy_and_exact",
        ),
        param(
            ["Categorical_1", "Switch_1", "Some_Setting"],
            id="exact_only",
        ),
        param(
            ["Num_disc_1", "Fraction_1"],
            id="fuzzy_only",
        ),
    ],
)
def test_fuzzy_anti_join(searchspace, dataframe_constructor):
    """Rows fuzzily matching the right dataframe are excluded from the result."""
    left_df = searchspace.discrete.get_candidates().copy()
    selected = np.random.choice(left_df.index, 4, replace=False)
    right_df = left_df.loc[selected].reset_index(drop=True)

    # Perturb within tolerance, so the rows still count as matches
    add_parameter_noise(
        right_df,
        searchspace.discrete.parameters,
        noise_type="absolute",
        noise_level=1e-6,
    )

    left = dataframe_constructor(left_df.to_dict(orient="list"))
    right = dataframe_constructor(right_df.to_dict(orient="list"))

    tolerances = {
        p.name: p.tolerance
        for p in searchspace.discrete.parameters
        if p.is_numerical and p.is_discrete
    }
    remaining = (
        _fuzzy_anti_join(left, right, tolerances)
        .collect()
        .to_pandas()
        .reset_index(drop=True)
    )

    def sort_rows(df: pd.DataFrame) -> pd.DataFrame:
        cols = list(df.columns)
        return df.sort_values(cols).reset_index(drop=True)

    expected = left_df.drop(index=selected).reset_index(drop=True)
    assert_frame_equal(
        sort_rows(remaining), sort_rows(expected), check_dtype=False, check_like=True
    )


@pytest.mark.parametrize(
    "parameter_names", [["Categorical_1", "Num_disc_1", "Some_Setting"]]
)
def test_fuzzy_anti_join_no_right_rows(searchspace):
    """An empty ``right`` dataframe excludes no rows of ``left``."""
    left_df = searchspace.discrete.get_candidates().copy()
    right_df = left_df.iloc[0:0]

    tolerances = {
        p.name: p.tolerance
        for p in searchspace.discrete.parameters
        if p.is_numerical and p.is_discrete
    }
    remaining = _fuzzy_anti_join(left_df, right_df, tolerances).collect()
    assert_frame_equal(remaining.to_pandas(), left_df)


def test_fuzzy_anti_join_no_common_columns():
    """If the two dataframes share no columns at all, an error is raised."""
    left_df = pd.DataFrame({"a": [1, 2]})
    right_df = pd.DataFrame({"z": [1]})

    with pytest.raises(ValueError, match="no columns in common"):
        _fuzzy_anti_join(left_df, right_df, {})


@pytest.mark.parametrize(
    "tolerances",
    [
        param({"d": 0.1}, id="tolerance_column_missing_everywhere"),
        param({"a": 0.1}, id="tolerance_column_missing_from_right"),
        param({"c": 0.1}, id="tolerance_column_missing_from_left"),
    ],
)
def test_fuzzy_anti_join_invalid_columns(tolerances):
    """Tolerance columns missing from ``left`` or ``right`` raise an error."""
    left_df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    right_df = pd.DataFrame({"b": [1], "c": [2]})
    with pytest.raises(ValueError, match="not shared by both dataframes"):
        _fuzzy_anti_join(left_df, right_df, tolerances)


def test_fuzzy_anti_join_extra_right_columns():
    """Columns present only in ``right`` are ignored."""
    left_df = pd.DataFrame({"a": [1, 2, 3]})
    right_df = pd.DataFrame({"a": [1], "z": [999]})

    remaining = _fuzzy_anti_join(left_df, right_df, {}).collect()
    assert sorted(remaining.to_pandas()["a"]) == [2, 3]


def test_measurement_singletons():
    """Correct treatment of rows with unmeasured targets."""
    df = pd.DataFrame(
        {
            "A": [np.nan, 2, 3, 4, 5],
            "B": ["a", "b", "b", "a", np.nan],
            "C": [55, 44, 33, 22, 11],
        },
        index=[123, 2, 3, 4, 5],
    )
    targets = [
        NumericalTarget(name="A"),
        BinaryTarget(name="B", success_value="a", failure_value="b"),
    ]

    # Test NaN block
    with pytest.raises(
        ValueError, match=r"Bad input in the rows with these indices: \[123, 5\]"
    ):
        handle_missing_values(df, [t.name for t in targets])

    # Test NaN removal
    df_new = handle_missing_values(df, [t.name for t in targets], drop=True)
    assert_frame_equal(df.iloc[1:-1, :], df_new)


@pytest.mark.parametrize(
    "data, warning, match",
    [
        param(
            pd.DataFrame(
                {
                    "Num_disc_1": [1.1, 2.2],
                    "Target_max": [3.3, 4.4],
                }
            ),
            None,
            "",
            id="valid_float",
        ),
        param(
            pd.DataFrame(
                {
                    "Num_disc_1": [1, 2],
                    "Target_max": [3, 4],
                }
            ),
            None,
            "",
            id="valid_int",
        ),
        param(
            pd.DataFrame(
                {
                    "Num_disc_1": [1.1, True],
                    "Target_max": [3.3, 4.4],
                }
            ),
            InputDataTypeWarning,
            "Num_disc_1",
            id="bool_in_num_parameter",
        ),
        param(
            pd.DataFrame(
                {
                    "Num_disc_1": [1.1, 2.2],
                    "Target_max": [3.3, False],
                }
            ),
            InputDataTypeWarning,
            "Target_max",
            id="bool_in_num_target",
        ),
    ],
)
def test_input_dtype(data, warning, match, parameters, targets):
    """If necessary, utility converts input dtype and raises warning."""
    with nullcontext() if warning is None else pytest.warns(warning, match=match):
        converted = normalize_input_dtypes(data, parameters + targets)
        converted = normalize_input_dtypes(data, parameters + targets)
        print(data)
        print(converted)

    # Asserts converted columns have expected dtypes
    assert pd.api.types.is_float_dtype(converted["Num_disc_1"]), (data, converted)
    assert pd.api.types.is_float_dtype(converted["Target_max"]), (data, converted)
