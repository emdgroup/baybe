"""Tests for dataframe utilities."""

from contextlib import nullcontext

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pytest import param

from baybe.exceptions import InputDataTypeWarning, SearchSpaceMatchWarning
from baybe.targets import BinaryTarget, NumericalTarget
from baybe.utils.dataframe import (
    add_noise_to_perturb_degenerate_rows,
    add_parameter_noise,
    fuzzy_row_match,
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
    ("parameter_names", "noise", "duplicated"),
    [
        param(
            ["Categorical_1", "Num_disc_1", "Some_Setting"],
            False,
            True,
            id="discrete_num_noiseless_duplicated",
        ),
        param(
            ["Categorical_1", "Num_disc_1", "Some_Setting"],
            False,
            False,
            id="discrete_num_noiseless_unique",
        ),
        param(
            ["Categorical_1", "Num_disc_1", "Some_Setting"],
            True,
            False,
            id="discrete_num_noisy_unique",
        ),
        param(
            ["Categorical_1", "Switch_1", "Some_Setting"],
            False,
            False,
            id="discrete_cat",
        ),
        param(
            ["Categorical_1", "Switch_1", "Conti_finite1"],
            False,
            False,
            id="hybrid_cat",
        ),
        param(
            ["Categorical_1", "Num_disc_1", "Conti_finite1"],
            False,
            False,
            id="hybrid_num_noiseless_unique",
        ),
        param(
            ["Categorical_1", "Num_disc_1", "Conti_finite1"],
            True,
            False,
            id="hybrid_num_noisy_unique",
        ),
        param(
            ["Categorical_1", "Num_disc_1", "Conti_finite1"],
            False,
            True,
            id="hybrid_num_noiseless_duplicated",
        ),
    ],
)
def test_fuzzy_row_match(searchspace, noise, duplicated):
    """Fuzzy row matching returns expected indices."""
    left_df = searchspace.discrete.exp_rep.copy()
    selected = np.random.choice(left_df.index, 4, replace=False)
    right_df = left_df.loc[selected].reset_index(drop=True)

    context = nullcontext()
    if duplicated:
        # Set one of the input values to exactly the midpoint between two values to
        # cause a degenerate match
        vals = searchspace.get_parameters_by_name(["Num_disc_1"])[0].values
        right_df.loc[0, "Num_disc_1"] = vals[0] + (vals[1] - vals[0]) / 2.0
        context = pytest.warns(SearchSpaceMatchWarning, match="multiple matches")

    if noise:
        add_parameter_noise(
            right_df,
            searchspace.discrete.parameters,
            noise_type="relative_percent",
            noise_level=0.1,
        )

    with context as c:
        matched = fuzzy_row_match(left_df, right_df, searchspace.parameters)

    if duplicated:
        # Assert correct identification of problematic df parts
        w = next(x for x in c if isinstance(x.message, SearchSpaceMatchWarning)).message
        assert_frame_equal(right_df.loc[[0]], w.data)

        # Ignore problematic indices for subsequent equality check
        selected = selected[1:]
        matched = matched[1:]

    assert set(selected) == set(matched), (selected, matched)


@pytest.mark.parametrize(
    "parameter_names",
    [
        param(["Categorical_1", "Categorical_2", "Switch_1"], id="discrete"),
        param(["Categorical_1", "Num_disc_1", "Conti_finite1"], id="hybrid"),
    ],
)
@pytest.mark.parametrize("invalid", ["left", "right"])
def test_invalid_fuzzy_row_match(searchspace, invalid):
    """Returns expected errors when dataframes don't contain all expected columns."""
    left_df = searchspace.discrete.exp_rep.copy()
    selected = np.random.choice(left_df.index, 4, replace=False)
    right_df = left_df.loc[selected].copy()

    # Drop first column
    if invalid == "left":
        left_df = left_df.iloc[:, 1:]
    else:
        right_df = right_df.iloc[:, 1:]

    match = f"corresponding column in the {invalid} dataframe."
    with pytest.raises(ValueError, match=match):
        fuzzy_row_match(left_df, right_df, searchspace.parameters)


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
            id="valid",
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
