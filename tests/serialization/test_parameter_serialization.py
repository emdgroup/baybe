"""Parameter serialization tests."""

import hypothesis.strategies as st
import pytest
from hypothesis import given
from pytest import param

from baybe._optional.info import CHEM_INSTALLED
from tests.hypothesis_strategies.parameters import (
    categorical_parameters,
    custom_parameters,
    numerical_continuous_parameters,
    numerical_discrete_parameters,
    substance_parameters,
    task_parameters,
)
from tests.serialization.utils import assert_roundtrip_consistency


@pytest.mark.parametrize(
    "strategy",
    [
        param(numerical_discrete_parameters(), id="NumericalDiscreteParameter"),
        param(numerical_continuous_parameters(), id="NumericalContinuousParameter"),
        param(categorical_parameters(), id="CategoricalParameter"),
        param(task_parameters(), id="TaskParameter"),
        param(custom_parameters(), id="CustomParameter"),
        param(
            substance_parameters(),
            id="SubstanceParameter",
            marks=pytest.mark.skipif(
                not CHEM_INSTALLED, reason="Optional chem dependency not installed."
            ),
        ),
    ],
)
@given(data=st.data())
def test_roundtrip(strategy: st.SearchStrategy, data: st.DataObject):
    """A serialization roundtrip yields an equivalent object."""
    parameter = data.draw(strategy)
    assert_roundtrip_consistency(parameter)
