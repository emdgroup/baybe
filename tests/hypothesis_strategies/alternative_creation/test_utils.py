"""Test alternative ways of creation not considered in the strategies."""

import pytest
from pytest import param

from baybe.utils.interval import Interval


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        param(None, 0, id="left-unbounded"),
        param(0, None, id="right-unbounded"),
        param(None, None, id="fully-unbounded"),
    ],
)
def test_none_bounds(lower, upper):
    """Bounds can also be None."""
    Interval(lower, upper)
