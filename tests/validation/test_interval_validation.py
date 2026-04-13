"""Validation tests for intervals."""

import pytest
from pytest import param

from baybe.utils.interval import Interval


@pytest.mark.parametrize(
    "bounds",
    [
        param((1.0, 0.0), id="wrong_order"),
        param(("a", 0.0), id="string"),
        param((float("nan"), 0.0), id="nan"),
    ],
)
def test_invalid_range(request, bounds):
    """Providing invalid bounds raises an exception."""
    with pytest.raises(ValueError):
        Interval(*bounds)

    # Also test the reverse order (except for cases where the order matters)
    if request.node.callspec.id == "wrong_order":
        return
    with pytest.raises(ValueError):
        Interval(*bounds[::-1])


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
    """Intervals that are close up to floating-point precision are detected."""
    assert (Interval(0, 1) == other) == expected
