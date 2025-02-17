"""Tests for dataframe utilities."""

import numpy as np
import pandas as pd
import pytest
from pytest import param

from baybe.utils.dataframe import (
    add_noise_to_perturb_degenerate_rows,
    add_parameter_noise,
    fuzzy_row_match,
)


@pytest.fixture()
def n_grid_points():
    return 5


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
        param(["Categorical_1", "Categorical_2", "Switch_1"], id="discrete"),
        param(["Categorical_1", "Num_disc_1", "Conti_finite1"], id="hybrid"),
    ],
)
@pytest.mark.parametrize("noise", [True, False], ids=["exact", "noisy"])
def test_fuzzy_row_match(searchspace, noise):
    """Test whether fuzzy row matching returns expected indices."""
    left_df = searchspace.discrete.exp_rep.copy()
    selected = np.random.choice(left_df.index, 4, replace=False)
    right_df = left_df.loc[selected].copy()

    if noise:
        add_parameter_noise(
            right_df,
            searchspace.discrete.parameters,
            noise_type="relative_percent",
            noise_level=0.1,
        )
    matched = fuzzy_row_match(left_df, right_df, searchspace.parameters)

    assert set(selected) == set(matched), (selected, matched)


@pytest.mark.parametrize(
    "parameter_names",
    [
        param(["Categorical_1", "Categorical_2", "Switch_1"], id="discrete"),
        param(["Categorical_1", "Num_disc_1", "Conti_finite1"], id="hybrid"),
    ],
)
@pytest.mark.parametrize("invalid", ["left_invalid", "right_invalid"])
def test_invalid_fuzzy_row_match(searchspace, invalid, n_grid_points):
    """Test whether fuzzy row matching returns expected errors."""
    left_df = searchspace.discrete.exp_rep.copy()
    selected = np.random.choice(left_df.index, 4, replace=False)
    right_df = left_df.loc[selected].copy()

    # Drop first column
    if invalid == "left_invalid":
        left_df = left_df.iloc[:, 1:]
    else:
        right_df = right_df.iloc[:, 1:]

    with pytest.raises(ValueError):
        fuzzy_row_match(left_df, right_df, searchspace.parameters)
