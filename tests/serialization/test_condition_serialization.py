"""Condition serialization tests."""

import hypothesis.strategies as st
import pytest
from hypothesis import given
from pytest import param

from tests.hypothesis_strategies.conditions import (
    sub_selection_conditions,
    threshold_conditions,
)
from tests.serialization.utils import assert_roundtrip_consistency


@pytest.mark.parametrize(
    "strategy",
    [
        param(threshold_conditions(), id="ThresholdCondition"),
        param(sub_selection_conditions(), id="SubSelectionCondition"),
    ],
)
@given(data=st.data())
def test_roundtrip(strategy: st.SearchStrategy, data: st.DataObject):
    """A serialization roundtrip yields an equivalent object."""
    condition = data.draw(strategy)
    assert_roundtrip_consistency(condition)
