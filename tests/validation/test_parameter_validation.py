"""Validation tests for parameters."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from attrs import NOTHING
from cattrs.errors import IterableValidationError
from pytest import param

from baybe._optional.info import CHEM_INSTALLED
from baybe.parameters.categorical import (
    CategoricalParameter,
    TaskParameter,
)
from baybe.parameters.custom import CustomDiscreteParameter
from baybe.parameters.numerical import (
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.parameters.substance import SubstanceParameter
from baybe.parameters.validation import validate_decorrelation
from baybe.utils.interval import InfiniteIntervalError

try:  # For python < 3.11, use the exceptiongroup backport
    ExceptionGroup
except NameError:
    from exceptiongroup import ExceptionGroup


@pytest.mark.parametrize(
    ("name", "error"),
    [
        param("", ValueError, id="empty_string"),
        param(1, TypeError, id="not_a_string"),
    ],
)
def test_invalid_parameter_name(name, error):
    """Providing an invalid parameter name raises an exception."""
    with pytest.raises(error):
        NumericalDiscreteParameter(name=name, values=[1, 2, 3])


@pytest.mark.parametrize(
    ("decorrelation", "error"),
    [
        param(1, TypeError, id="wrong_type"),
        param(1.0, ValueError, id="too_high"),
        param(0.0, ValueError, id="too_low"),
        param(float("nan"), ValueError, id="nan"),
        param(float("inf"), ValueError, id="inf"),
    ],
)
def test_invalid_decorrelation(decorrelation, error):
    """Providing an invalid decorrelation setting raises an exception."""
    with pytest.raises(error):
        validate_decorrelation(Mock(), Mock(), decorrelation)


@pytest.mark.parametrize(
    ("values", "error"),
    [
        param([1, np.inf], ValueError, id="infinity"),
        param([1, np.nan], ValueError, id="nan"),
        param([1, 1], ValueError, id="duplicates"),
        param([1], ValueError, id="only_one_value"),
        param([1, "a"], IterableValidationError, id="not_a_number"),
    ],
)
def test_invalid_values_numerical_discrete_parameter(values, error):
    """Providing invalid parameter values raises an exception."""
    with pytest.raises(error):
        NumericalDiscreteParameter(name="invalid_values", values=values)


@pytest.mark.parametrize(
    ("bounds", "error"),
    [
        param([-np.inf, np.inf], InfiniteIntervalError, id="infinite"),
        param([0, np.inf], InfiniteIntervalError, id="open_right"),
        param([-np.inf, 0], InfiniteIntervalError, id="open_left"),
        param([0, 0], ValueError, id="degenerate"),
    ],
)
def test_invalid_bounds_numerical_continuous_parameter(bounds, error):
    """Creating an unbounded parameter raises an exception."""
    with pytest.raises(error):
        NumericalContinuousParameter(name="invalid_values", bounds=bounds)


@pytest.mark.parametrize(
    ("values", "error"),
    [
        param("ABC", ValueError, id="string"),
        param(["", "A"], ValueError, id="empty_string"),
        param(["A", "A"], ValueError, id="duplicates"),
        param(["A"], ValueError, id="only_one_value"),
        param(["A", 1], TypeError, id="not_a_string"),
    ],
)
def test_invalid_values_categorical_parameter(values, error):
    """Providing invalid parameter values raises an exception."""
    with pytest.raises(error):
        CategoricalParameter(name="invalid_values", values=values)


def test_invalid_encoding_categorical_parameter():
    """Providing an invalid encoding raises an exception."""
    with pytest.raises(ValueError):
        CategoricalParameter(name="invalid_encoding", values=["A", "B"], encoding="enc")


@pytest.mark.parametrize(
    ("values", "active_values", "error"),
    [
        param("ABC", NOTHING, ValueError, id="string"),
        param(["", "A"], NOTHING, ValueError, id="empty_string"),
        param(["A", "A"], NOTHING, ValueError, id="duplicates"),
        param(["A"], NOTHING, ValueError, id="only_one_value"),
        param(["A", 1], NOTHING, TypeError, id="not_a_string"),
        param(["A", "B"], [], ValueError, id="no_active_values"),
        param(["A", "B"], ["C"], ValueError, id="unknown_active_values"),
        param(["A", "B"], ["A", "A"], ValueError, id="duplicate_active_values"),
    ],
)
def test_invalid_values_task_parameter(values, active_values, error):
    with pytest.raises(error):
        """Providing invalid (active) parameter values raises an exception."""
        TaskParameter(name="invalid_values", values=values, active_values=active_values)


@pytest.mark.skipif(
    not CHEM_INSTALLED, reason="Optional chem dependency not installed."
)
@pytest.mark.parametrize(
    ("data", "error"),
    [
        param({"": "C", "A": "C"}, ValueError, id="empty_string"),
        param({"A": "C"}, ValueError, id="only_one_value"),
        param({"A": "C", 1: "C"}, TypeError, id="not_a_string"),
        param({"A": "C", "B": "X", "C": "Y"}, ExceptionGroup, id="invalid_smiles"),
        param(
            {"A": "CC", "B": "C-C", "C": "CCO", "D": "OCC"},
            ExceptionGroup,
            id="duplicate_substances",
        ),
    ],
)
def test_invalid_data_substance_parameter(data, error):
    """Providing invalid substance data raises an exception."""
    with pytest.raises(error):
        SubstanceParameter(name="invalid_data", data=data)


@pytest.mark.skipif(
    not CHEM_INSTALLED, reason="Optional chem dependency not installed."
)
def test_invalid_encoding_substance_parameter():
    """Providing an invalid encoding raises an exception."""
    with pytest.raises(ValueError):
        SubstanceParameter(
            name="invalid_encoding", data={"A": "C", "B": "CC"}, encoding="enc"
        )


@pytest.mark.parametrize(
    "data",
    [
        param(pd.DataFrame([[1, 2], [3, np.nan]], index=["A", "B"]), id="nan"),
        param(pd.DataFrame([[1, 2], [3, np.inf]], index=["A", "B"]), id="infinity"),
        param(pd.DataFrame([[1, 2], [3, 4]], index=["A", "A"]), id="duplicates"),
        param(pd.DataFrame([[1, 2]], index=["A"]), id="only_one_value"),
        param(pd.DataFrame([[1, 2], [1, 2]], index=["A", "B"]), id="zero_variance"),
        param(pd.DataFrame([[1, 2], [3, "a"]], index=["A", "B"]), id="wrong_type"),
        param(pd.DataFrame([[1, 2], [3, 4]], index=["A", 1]), id="not_a_string"),
        param(pd.DataFrame([[1, 2], [3, 4]], index=["A", ""]), id="empty_string"),
    ],
)
def test_invalid_data_custom_parameter(data):
    """Providing an invalid custom encoding raises an exception."""
    with pytest.raises(ValueError):
        CustomDiscreteParameter(name="invalid_data", data=data)
