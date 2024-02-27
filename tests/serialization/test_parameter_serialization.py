"""Test serialization of parameters."""

import hypothesis.strategies as st
import pytest
from hypothesis import given
from pytest import param

from baybe.parameters.base import Parameter

from ..conftest import _CHEM_INSTALLED
from ..hypothesis_strategies.parameters import (
    categorical_parameter,
    custom_parameter,
    numerical_continuous_parameter,
    numerical_discrete_parameter,
    substance_parameter,
    task_parameter,
)


@pytest.mark.parametrize(
    "parameter_strategy",
    [
        param(numerical_discrete_parameter(), id="NumericalDiscreteParameter"),
        param(numerical_continuous_parameter(), id="NumericalContinuousParameter"),
        param(categorical_parameter(), id="CategoricalParameter"),
        param(task_parameter(), id="TaskParameter"),
        param(custom_parameter(), id="CustomParameter"),
        param(
            substance_parameter(),
            id="SubstanceParameter",
            marks=pytest.mark.skipif(
                not _CHEM_INSTALLED, reason="Optional chem dependency not installed."
            ),
        ),
    ],
)
@given(data=st.data())
def test_parameter_roundtrip(parameter_strategy, data):
    """A serialization roundtrip yields an equivalent object."""
    parameter = data.draw(parameter_strategy)
    string = parameter.to_json()
    parameter2 = Parameter.from_json(string)
    assert parameter == parameter2, (parameter, parameter2)
