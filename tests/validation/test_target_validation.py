"""Validation tests for targets."""

import pytest
from pytest import param

from baybe.targets.numerical import NumericalTarget


@pytest.mark.parametrize(
    ("mode", "bounds"),
    [
        param("MATCH", None, id="non_closed_match_mode"),
        param("MAX", (0, None), id="half_open"),
    ],
)
def test_invalid_bounds_mode(mode, bounds):
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
def test_incompatible_transform_mode(mode, bounds, transformation):
    with pytest.raises(ValueError):
        NumericalTarget(
            name="incompatible_transform",
            mode=mode,
            bounds=bounds,
            transformation=transformation,
        )
