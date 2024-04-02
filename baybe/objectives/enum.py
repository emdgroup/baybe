"""Objective-related enumerations."""

from enum import Enum


class Scalarization(Enum):
    """Available scalarization mechanisms for desirability objectives."""

    MEAN = "MEAN"
    """Arithmetic mean."""

    GEOM_MEAN = "GEOM_MEAN"
    """Geometric mean."""
