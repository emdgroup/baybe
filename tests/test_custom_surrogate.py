"""Tests for custom surrogate models."""
from contextlib import nullcontext

import pytest

from baybe import Campaign
from baybe.exceptions import ModelParamsNotSupportedError
from baybe.surrogates import _ONNX_INSTALLED, register_custom_architecture
from tests.conftest import run_iterations

if _ONNX_INSTALLED:
    from baybe.surrogates import CustomONNXSurrogate


@pytest.mark.skipif(
    not _ONNX_INSTALLED, reason="Optional onnx dependency not installed."
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

    # Scenario: Model Params non-empty
    with pytest.raises(ModelParamsNotSupportedError):
        CustomONNXSurrogate(
            onnx_input_name="input",
            onnx_str=onnx_str,
            model_params={"Non_empty_dict": None},
        )


@pytest.mark.skipif(
    not _ONNX_INSTALLED, reason="Optional onnx dependency not installed."
)
def test_invalid_onnx_str():
    """Invalid onnx string causes error."""
    with pytest.raises(Exception):
        CustomONNXSurrogate(onnx_input_name="input", onnx_str=b"onnx_str")


@pytest.mark.skipif(
    not _ONNX_INSTALLED, reason="Optional onnx dependency not installed."
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


def test_validate_architectures():
    """Test architecture class validation."""
    # Scenario: Empty Class
    with pytest.raises(ValueError):
        register_custom_architecture()(type("EmptyArch"))

    # Scenario: Class with just `_fit`
    with pytest.raises(ValueError):
        register_custom_architecture()(type("PartialArch", (), {"_fit": True}))

    # Scenario: Class with `_fit` and `_posterior` but not methods
    with pytest.raises(ValueError):
        register_custom_architecture()(
            type("PartialArch", (), {"_fit": True, "_posterior": True})
        )

    # Scenario: Class with invalid `_fit` and `_posterior` methods
    def _invalid_func(invalid_param1, invalid_param2=1):
        return invalid_param1 + invalid_param2

    with pytest.raises(ValueError):
        register_custom_architecture()(
            type(
                "InvalidArch", (), {"_fit": _invalid_func, "_posterior": _invalid_func}
            )
        )

    # Scenario: Class with valid `_fit` but invalid `_posterior` methods
    def _valid_fit(self, searchspace, train_x, train_y):
        return self and searchspace and train_x and train_y

    with pytest.raises(ValueError):
        register_custom_architecture()(
            type("InvalidArch", (), {"_fit": _valid_fit, "_posterior": _invalid_func})
        )

    # Scenario: Both Valid
    def _valid_posterior(self, candidates):
        return self and candidates

    register_custom_architecture()(
        type("ValidArch", (), {"_fit": _valid_fit, "_posterior": _valid_posterior})
    )
