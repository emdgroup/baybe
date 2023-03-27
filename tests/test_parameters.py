"""Tests for the parameters module."""

import numpy as np
import pandas as pd
import pytest

from baybe.parameters import (
    Categorical,
    Custom,
    GenericSubstance,
    NumericContinuous,
    NumericDiscrete,
)
from baybe.utils import StrictValidationError
from pydantic import ValidationError

# Validation errors to be caught
validation_errors = (ValidationError, StrictValidationError)


def test_invalid_parameter_creation():
    """Invalid parameter creation raises expected error."""

    # Scenario: discrete numerical parameter contains duplicated values
    with pytest.raises(validation_errors):
        NumericDiscrete(
            name="num_duplicated",
            values=[1, 2, 3, 2],
        )

    # Scenario: categorical parameter contains duplicated values
    with pytest.raises(validation_errors):
        Categorical(
            name="cat_duplicated",
            values=["very bad", "bad", "OK", "OK"],
        )

    # Scenario: substance parameter contains invalid SMILES
    with pytest.raises(validation_errors):
        GenericSubstance(
            name="substance_invalid_smiles",
            data={"valid1": "C", "valid2": "CC", "invalid": "cc"},
        )

    # Scenario: custom parameter contains duplicated index
    with pytest.raises(validation_errors):
        Custom(
            name="custom_duplicated_index",
            data=pd.DataFrame(
                {
                    "D1": [1.1, 1.4, 1.7, 0.8],
                    "D2": [11, 23, 55, 23],
                    "D3": [-4, -13, 4, -2],
                },
                index=["mol1", "mol2", "mol3", "mol1"],
            ),
        )

    # Scenario: continuous numerical parameter has invalid bounds
    with pytest.raises(validation_errors):
        NumericContinuous(
            name="conti_invalid_bounds",
            bounds=(1, 0),
        )

    # Scenario: continuous numerical parameter has invalid bounds
    with pytest.raises(validation_errors):
        NumericContinuous(
            name="conti_invalid_bounds",
            bounds=(np.inf, 1),
        )

    # Scenario: continuous numerical parameter has invalid bounds
    with pytest.raises(validation_errors):
        NumericContinuous(
            name="conti_invalid_bounds",
            bounds=(1, 1),
        )
