"""Tests for custom surrogate models."""

from contextlib import nullcontext

import pytest

from baybe import Campaign
from baybe._optional.info import ONNX_INSTALLED
from baybe.surrogates import CustomONNXSurrogate
from tests.conftest import run_iterations


@pytest.mark.skipif(
    not ONNX_INSTALLED, reason="Optional onnx dependency not installed."
)
def test_invalid_onnx_creation(onnx_str):
    """Invalid onnx model creation."""
    # Scenario: No input
    with pytest.raises(TypeError):
        CustomONNXSurrogate()

    # Scenario: No onnx input name
    with pytest.raises(TypeError):
        CustomONNXSurrogate(onnx_str=b"onnx_str")

    # Scenario: No onnx str
    with pytest.raises(TypeError):
        CustomONNXSurrogate(onnx_input_name="input")


@pytest.mark.skipif(
    not ONNX_INSTALLED, reason="Optional onnx dependency not installed."
)
def test_invalid_onnx_str():
    """Invalid onnx string causes error."""
    with pytest.raises(Exception):
        CustomONNXSurrogate(onnx_input_name="input", onnx_str=b"onnx_str")


@pytest.mark.skipif(
    not ONNX_INSTALLED, reason="Optional onnx dependency not installed."
)
@pytest.mark.parametrize("surrogate_model", ["onnx"], indirect=True)
@pytest.mark.parametrize(
    ["parameter_names", "should_raise"],
    [(["Categorical_1"], True), (["SomeSetting"], False)],
)
def test_supported_parameter_types(campaign: Campaign, should_raise: bool):
    """Using an ONNX model with unsupported parameters should raise an exception."""
    run_iterations(campaign, n_iterations=1, batch_size=1)
    context = pytest.raises(TypeError) if should_raise else nullcontext()
    with context:
        campaign.recommend(batch_size=1)
