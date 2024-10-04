"""Validation tests for targets."""

import pytest
from pytest import param

from baybe.targets.binary import BinaryTarget
from baybe.targets.numerical import NumericalTarget


@pytest.mark.parametrize(
    ("mode", "bounds"),
    [
        param("MATCH", None, id="non_closed_match_mode"),
        param("MAX", (0, None), id="half_open"),
        param("MAX", (0, 0), id="degenerate"),
    ],
)
def test_numerical_target_invalid_bounds_mode(mode, bounds):
    """Providing invalid bounds raises an exception."""
    with pytest.raises(ValueError):
        NumericalTarget(name="invalid_bounds", mode=mode, bounds=bounds)


@pytest.mark.parametrize(
    ("mode", "bounds", "transformation"),
    [
        param("MIN", None, "BELL", id="bell_for_min"),
        param("MATCH", (0, 1), "LINEAR", id="linear_for_match"),
    ],
)
def test_numerical_target_incompatible_transform_mode(mode, bounds, transformation):
    with pytest.raises(ValueError):
        NumericalTarget(
            name="incompatible_transform",
            mode=mode,
            bounds=bounds,
            transformation=transformation,
        )


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
