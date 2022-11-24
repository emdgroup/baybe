"""
Tests for basic input-output nad iterative loop.
"""
import pandas as pd
import pytest

from baybe.core import BayBEConfig

# Dictionary containing items describing config tests that should throw an error.
# Key is a string describing the test and is displayed by pytest. Each value is a pair
# of the first item being the config dictionary update that is done to the default
# fixture and the second item being the expected exception type.
from baybe.utils import StrictValidationError
from pydantic import ValidationError

invalid_tests = {
    "duplicated_num_parameter": (
        {
            "parameters": [
                {
                    "name": "Numerical_New",
                    "type": "NUM_DISCRETE",
                    "values": [1, 2, 3, 2],
                }
            ],
        },
        ValidationError,
    ),
    "duplicated_cat_parameter": (
        {
            "parameters": [
                {
                    "name": "Categorical_New",
                    "type": "CAT",
                    "values": ["very bad", "bad", "OK", "OK"],
                }
            ],
        },
        ValidationError,
    ),
    "empty_parameter_list": (
        {
            "parameters": [],
        },
        ValidationError,
    ),
    "empty_target_list": (
        {
            "targets": [],
        },
        ValidationError,
    ),
    "substance_invalid_SMILES": (
        {
            "parameters": [
                {
                    "name": "Invalid_Substance",
                    "type": "SUBSTANCE",
                    "data": {"valid1": "C", "valid2": "CC", "invalid": "cc"},
                }
            ],
        },
        StrictValidationError,
    ),
    "substance_invalid_field": (
        {
            "parameters": [
                {
                    "name": "Substance_New",
                    "type": "SUBSTANCE",
                    # Substance parameter does not use 'values' but 'data' field
                    "values": {"water": "O", "methane": "C"},
                }
            ],
        },
        ValidationError,
    ),
    "custom_duplicated_index": (
        {
            "parameters": [
                {
                    "name": "Custom_1",
                    "type": "CUSTOM",
                    "data": pd.DataFrame(
                        {
                            "D1": [1.1, 1.4, 1.7, 0.8],
                            "D2": [11, 23, 55, 23],
                            "D3": [-4, -13, 4, -2],
                        },
                        index=["mol1", "mol2", "mol3", "mol1"],
                    ),
                },
            ],
        },
        StrictValidationError,
    ),
}


@pytest.mark.xfail
@pytest.mark.parametrize("config_update_key", invalid_tests.keys())
def test_invalid_config(config_basic_1target, config_update_key):
    """
    Ensure invalid configurations trigger defined exceptions.
    """
    config_update, expected_error = invalid_tests[config_update_key]
    config_basic_1target.update(config_update)

    with pytest.raises(expected_error):
        BayBEConfig(**config_basic_1target)
