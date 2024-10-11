"""Tests for the custom parameter."""

import numpy as np
import pandas as pd
import pytest
from pytest import param

from baybe.parameters import CustomDiscreteParameter

from .conftest import run_iterations


@pytest.mark.parametrize("parameter_names", [["Custom_1", "Custom_2"]])
def test_run_iterations(campaign, n_iterations, batch_size):
    """Test if iterative loop runs with custom parameters."""
    run_iterations(campaign, n_iterations, batch_size)


@pytest.mark.parametrize(
    "data, msg",
    [
        param(
            pd.DataFrame([[1, 2], [3, "A"]], index=["a", "b"]),
            "contains non-numeric values",
            id="non_numeric_values",
        ),
        param(
            pd.DataFrame([[1, 2], [3, 4]], index=["a", 1]),
            "contains non-string index values",
            id="non_string_index",
        ),
        param(
            pd.DataFrame([[1, 2], [3, 4]], index=["a", ""]),
            "contains empty string index values",
            id="empty_string_index",
        ),
        param(
            pd.DataFrame([[1, 2], [3, np.inf]], index=["a", "b"]),
            "contains nan/infinity entries",
            id="contains_inf",
        ),
        param(
            pd.DataFrame([[1, 2], [3, np.nan]], index=["a", "b"]),
            "contains nan/infinity entries",
            id="contains_nan",
        ),
        param(
            pd.DataFrame([[1, 2], [3, 4]], index=["a", "a"]),
            "contains duplicated indices",
            id="duplicate_indices",
        ),
        param(
            pd.DataFrame([[1, 2], [3, 2]], index=["a", "b"]),
            "columns that contain only a single value",
            id="constant_column",
        ),
        param(
            pd.DataFrame([[1, 2], [1, 2], [3, 4]], index=["a", "b", "c"]),
            "ensure all labels have a unique numerical representation",
            id="duplicate_rows",
        ),
    ],
)
def test_invalid_input(data, msg):
    """Test if invalid duplicated rows are detected."""
    with pytest.raises(ValueError, match=msg):
        CustomDiscreteParameter(name="p", data=data)
