"""Validation tests for intervals."""

import pytest
from pytest import param

from baybe.utils.interval import Interval


@pytest.mark.parametrize(
    "bounds",
    [
        param((1.0, 0.0), id="wrong_bounds_order"),
    ],
)
def test_invalid_range(bounds):
    """Providing a non-increasing pair of floats raises an exception."""
    with pytest.raises(ValueError):
        Interval(*bounds)


@pytest.mark.parametrize(
    "bounds",
    [
        param(("a", 0.0), id="string"),
    ],
)
def test_invalid_types(bounds):
    """Providing an invalid bound type raises an exception."""
    with pytest.raises(ValueError):
        Interval(*bounds)
