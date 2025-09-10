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
