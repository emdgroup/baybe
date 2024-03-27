"""Objective-related enumerations."""

from enum import Enum


class CombineFunc(Enum):
    """Available combine functions for desirability objectives."""

    MEAN = "MEAN"
    """Arithmetic mean."""

    GEOM_MEAN = "GEOM_MEAN"
    """Geometric mean."""
