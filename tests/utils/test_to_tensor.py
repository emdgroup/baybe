"""Tests for the to_tensor utility."""

import numpy as np
import pandas as pd
import pytest
import torch
from pytest import param

from baybe.settings import active_settings
from baybe.utils.dataframe import to_tensor


@pytest.fixture(
    name="torch_dtype",
    autouse=True,
    params=[param(False, id="float64"), param(True, id="float32")],
)
def fixture_torch_dtype(request, monkeypatch):
    """Run every test under all torch float precisions."""
    monkeypatch.setattr(active_settings, "use_single_precision_torch", request.param)


# -------------------------------------------------------------------------------------
# Shared test data
# -------------------------------------------------------------------------------------

_MATRIX_DATA = [[1.0, 3.0], [2.0, 4.0]]
_FRAME_DATA = {"a": [1.0, 2.0], "b": [3.0, 4.0]}
_SERIES_DATA = [1.0, 2.0, 3.0]


def _ref_2d() -> torch.Tensor:
    """Expected tensor for _FRAME_DATA: shape (2, 2), row-major."""
    return torch.tensor(_MATRIX_DATA, dtype=active_settings.DTypeFloatTorch)


def _ref_1d() -> torch.Tensor:
    """Expected tensor for _SERIES_DATA: shape (3,)."""
    return torch.tensor(_SERIES_DATA, dtype=active_settings.DTypeFloatTorch)


# -------------------------------------------------------------------------------------
# Scalars
# -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        param(1, 1.0, id="int"),
        param(2.5, 2.5, id="float"),
    ],
)
def test_to_tensor_scalar(value, expected):
    """to_tensor converts scalars to 0-D tensors of the configured dtype."""
    result = to_tensor(value)
    assert result.dtype == active_settings.DTypeFloatTorch
    assert result.item() == expected


# -------------------------------------------------------------------------------------
# Numpy arrays
# -------------------------------------------------------------------------------------


def test_to_tensor_ndarray_2d():
    """to_tensor converts a 2-D numpy array."""
    array = np.array(_MATRIX_DATA)
    assert torch.allclose(to_tensor(array), _ref_2d())


def test_to_tensor_ndarray_1d():
    """to_tensor converts a 1-D numpy array."""
    array = np.array(_SERIES_DATA)
    assert torch.allclose(to_tensor(array), _ref_1d())


def test_to_tensor_ndarray_negative_strides():
    """to_tensor handles numpy arrays with negative strides."""
    array = np.array(_SERIES_DATA)[::-1]
    assert any(stride < 0 for stride in array.strides)
    result = to_tensor(array)
    assert torch.allclose(result, _ref_1d().flip(0))
    assert result.is_contiguous()


def test_to_tensor_ndarray_readonly():
    """to_tensor handles read-only numpy arrays."""
    array = np.array(_SERIES_DATA)
    array.flags.writeable = False
    assert torch.allclose(to_tensor(array), _ref_1d())


# -------------------------------------------------------------------------------------
# Pandas
# -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "expected_values"),
    [
        param(_FRAME_DATA, None, id="standard"),
        param(
            {"a": [True, False], "b": [4, 5]},
            [[1.0, 4.0], [0.0, 5.0]],
            id="mixed-bool-int",
        ),
    ],
)
def test_to_tensor_pandas_dataframe(data, expected_values):
    """to_tensor converts pandas DataFrames, including mixed-dtype columns."""
    result = to_tensor(pd.DataFrame(data))
    reference = (
        _ref_2d()
        if expected_values is None
        else torch.tensor(expected_values, dtype=active_settings.DTypeFloatTorch)
    )
    assert result.shape == reference.shape
    assert torch.allclose(result, reference)


def test_to_tensor_pandas_dataframe_negative_strides():
    """to_tensor handles pandas DataFrames backed by negatively-strided arrays."""
    series = pd.Series(_SERIES_DATA, name="x")
    dataframe = series[::-1].to_frame()
    assert any(stride < 0 for stride in dataframe.to_numpy().strides)
    result = to_tensor(dataframe)
    assert torch.allclose(result, _ref_1d().flip(0).unsqueeze(1))
    assert result.is_contiguous()


def test_to_tensor_pandas_series():
    """to_tensor converts a pandas Series."""
    series = pd.Series(_SERIES_DATA, name="x")
    assert torch.allclose(to_tensor(series), _ref_1d())


# -------------------------------------------------------------------------------------
# Variadic form
# -------------------------------------------------------------------------------------


def test_to_tensor_variadic():
    """to_tensor variadic form returns a tuple of tensors."""
    tensor_2d, tensor_1d = to_tensor(
        pd.DataFrame(_FRAME_DATA), pd.Series(_SERIES_DATA, name="x")
    )
    assert torch.allclose(tensor_2d, _ref_2d())
    assert torch.allclose(tensor_1d, _ref_1d())
