"""Tests for dataframe utilities."""

import numpy as np
import pandas as pd
import pytest

from baybe.utils.dataframe import add_noise_to_perturb_degenerate_rows


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
