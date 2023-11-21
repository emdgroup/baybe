"""Tests for the parameters module."""

import numpy as np
import pandas as pd
import pytest

from baybe.parameters import (
    CategoricalParameter,
    CustomDiscreteParameter,
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.utils.chemistry import _MORDRED_INSTALLED, _RDKIT_INSTALLED

_CHEM_INSTALLED = _MORDRED_INSTALLED and _RDKIT_INSTALLED
if _CHEM_INSTALLED:
    from baybe.parameters import SubstanceParameter


def test_invalid_parameter_creation():
    """Invalid parameter creation raises expected error."""
    # Scenario: discrete numerical parameter contains duplicated values
    with pytest.raises(ValueError):
        NumericalDiscreteParameter(
            name="num_duplicated",
            values=[1, 2, 3, 2],
        )

    # Scenario: categorical parameter contains duplicated values
    with pytest.raises(ValueError):
        CategoricalParameter(
            name="cat_duplicated",
            values=["very bad", "bad", "OK", "OK"],
        )

    # Scenario: substance parameter contains invalid SMILES
    if _CHEM_INSTALLED:
        with pytest.raises(ValueError):
            SubstanceParameter(
                name="substance_invalid_smiles",
                data={"valid1": "C", "valid2": "CC", "invalid": "cc"},
            )

    # Scenario: custom parameter contains duplicated index
    with pytest.raises(ValueError):
        CustomDiscreteParameter(
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
    with pytest.raises(ValueError):
        NumericalContinuousParameter(
            name="conti_invalid_bounds",
            bounds=(1, 0),
        )

    # Scenario: continuous numerical parameter has invalid bounds
    with pytest.raises(ValueError):
        NumericalContinuousParameter(
            name="conti_invalid_bounds",
            bounds=(np.inf, 1),
        )

    # Scenario: continuous numerical parameter has invalid bounds
    with pytest.raises(ValueError):
        NumericalContinuousParameter(
            name="conti_invalid_bounds",
            bounds=(1, 1),
        )
