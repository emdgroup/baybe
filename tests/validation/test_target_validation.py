"""Validation tests for targets."""

import pytest
from pytest import param

from baybe.targets.binary import BinaryTarget


@pytest.mark.parametrize(
    ("choices", "error", "match"),
    [
        param((None, 0), TypeError, "'success_value' must be", id="wrong_type"),
        param((0, 0), ValueError, "must be different", id="identical"),
    ],
)
def test_binary_target_invalid_values(choices, error, match):
    """Providing invalid choice values raises an error."""
    with pytest.raises(error, match=match):
        BinaryTarget(
            name="invalid_value",
            success_value=choices[0],
            failure_value=choices[1],
        )
