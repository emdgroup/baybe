"""Tests for custom surrogate models."""

import pytest
from baybe.surrogate import CustomONNXSurrogate
from baybe.utils.dataframe import add_fake_results


def test_invalid_onnx_creation():
    """Invalid onnx model creation."""
    # Scenario: No input
    with pytest.raises(TypeError):
        CustomONNXSurrogate()

    # Scenario: No onnx input name
    with pytest.raises(TypeError):
        CustomONNXSurrogate(onnx_str="onnx_str")

    # Scenario: No onnx str
    with pytest.raises(TypeError):
        CustomONNXSurrogate(onnx_input_name="input")

    # Scenario: Model Params non-empty
    with pytest.raises(ValueError):
        CustomONNXSurrogate(
            onnx_input_name="input",
            onnx_str="onnx_str",
            model_params={"Non_empty_dict": True},
        )


@pytest.mark.slow
@pytest.mark.parametrize(
    "surrogate_model",
    [CustomONNXSurrogate(onnx_input_name="input", onnx_str="onnx_str")],
)
def test_invalid_onnx_str(baybe):
    """Invalid onnx string causes error during `fit`."""
    rec = baybe.recommend()
    add_fake_results(rec, baybe)
    baybe.add_measurements(rec)

    with pytest.raises(ValueError):
        baybe.recommend()
