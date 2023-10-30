"""Tests for custom surrogate models."""

import pytest

from baybe.exceptions import ModelParamsNotSupportedError
from baybe.surrogate import _ONNX_INSTALLED, register_custom_architecture

if _ONNX_INSTALLED:
    from baybe.surrogate import CustomONNXSurrogate

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

        # Scenario: Model Params non-empty
        with pytest.raises(ModelParamsNotSupportedError):
            CustomONNXSurrogate(
                onnx_input_name="input",
                onnx_str=onnx_str,
                model_params={"Non_empty_dict": None},
            )

    def test_invalid_onnx_str():
        """Invalid onnx string causes error."""
        with pytest.raises(Exception):
            CustomONNXSurrogate(onnx_input_name="input", onnx_str=b"onnx_str")


def test_validate_architectures():
    """Test architecture class validation."""
    # Scenario: Empty Class
    with pytest.raises(ValueError) as excinfo:
        register_custom_architecture()(type("EmptyArch"))
        assert "must exist" in excinfo.value.message

    # Scenario: Class with just `_fit`
    with pytest.raises(ValueError):
        register_custom_architecture()(type("PartialArch", (), {"_fit": True}))
        assert "must exist" in excinfo.value.message

    # Scenario: Class with `_fit` and `_posterior` but not methods
    with pytest.raises(ValueError):
        register_custom_architecture()(
            type("PartialArch", (), {"_fit": True, "_posterior": True})
        )
        assert "must be methods" in excinfo.value.message

    # Scenario: Class with invalid `_fit` and `_posterior` methods
    def _invalid_func(invalid_param1, invalid_param2=1):
        return invalid_param1 + invalid_param2

    with pytest.raises(ValueError):
        register_custom_architecture()(
            type(
                "InvalidArch", (), {"_fit": _invalid_func, "_posterior": _invalid_func}
            )
        )
        assert "Invalid Arguments" in excinfo.value.message

    # Scenario: Class with valid `_fit` but invalid `_posterior` methods
    def _valid_fit(self, searchspace, train_x, train_y):
        return self and searchspace and train_x and train_y

    with pytest.raises(ValueError):
        register_custom_architecture()(
            type("InvalidArch", (), {"_fit": _valid_fit, "_posterior": _invalid_func})
        )
        assert "Invalid Arguments" in excinfo.value.message

    # Scenario: Both Valid
    def _valid_posterior(self, candidates):
        return self and candidates

    register_custom_architecture()(
        type("ValidArch", (), {"_fit": _valid_fit, "_posterior": _valid_posterior})
    )
