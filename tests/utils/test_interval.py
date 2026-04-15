"""Tests for interval utilities."""

import pytest
from pytest import param

from baybe.utils.interval import Interval


@pytest.mark.parametrize(
    ("other", "expected"),
    [
        param(Interval(0, 1), True, id="exact_match"),
        param(Interval(0, 0.9999999999999999), True, id="upper_float_imprecision"),
        param(Interval(1e-16, 1 - 1e-16), True, id="both_float_imprecision"),
        param(Interval(0, 0.5), False, id="different_upper"),
        param(Interval(0.5, 1), False, id="different_lower"),
    ],
)
def test_close_interval_bounds(other, expected):
    """Intervals with bounds close up to floating-point precision compare equal."""
    assert (Interval(0, 1) == other) == expected
