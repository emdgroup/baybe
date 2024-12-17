"""Validation tests for partitions."""

import pytest
from pytest import param

from baybe.utils.interval import Partition


@pytest.mark.parametrize(
    ("thresholds", "error", "match"),
    [
        param(1, TypeError, None, id="not_iterable"),
        param([], ValueError, None, id="too_short"),
        param([1, 0], ValueError, "monotonically increasing", id="decreasing"),
    ],
)
def test_invalid_thresholds(thresholds, error, match):
    """Providing invalid thresholds raises an error."""
    with pytest.raises(error, match=match):
        Partition(thresholds)
