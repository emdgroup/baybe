"""Test alternative ways of creation not considered in the strategies."""

import pytest
from pytest import param

from baybe.targets.enum import TargetMode, TargetTransformation
from baybe.targets.numerical import NumericalTarget


@pytest.mark.parametrize(
    "bounds",
    [
        param((None, None), id="unbounded"),
        param((0, 1), id="bounded"),
    ],
)
def test_unspecified_transformation(bounds):
    """The transformation can be unspecified, in which case a default is chosen.

    This explicitly tests the logic of the attrs default method.
    """
    NumericalTarget("unspecified", mode="MAX", bounds=bounds)


@pytest.mark.parametrize("mode", (m.name for m in TargetMode))
def test_string_mode(mode):
    """The mode can also be specified as a string instead of an enum value."""
    NumericalTarget("string_mode", mode=mode, bounds=(0, 1))


@pytest.mark.parametrize("transformation", (t.name for t in TargetTransformation))
def test_string_transformation(transformation):
    """The transformation can also be specified as a string instead of an enum value."""
    mode = "MAX" if transformation == "LINEAR" else "MATCH"
    NumericalTarget(
        "string_mode", mode=mode, bounds=(0, 1), transformation=transformation
    )
