"""Target-related enumerations."""

from enum import Enum


class TargetMode(Enum):
    """Available modes for targets."""

    MIN = "MIN"
    """The target is to be minimized."""

    MAX = "MAX"
    """The target is to be maximized."""

    MATCH = "MATCH"
    """The target should be close to a given value."""


class TargetTransformation(Enum):
    """Available target transformations."""

    LINEAR = "LINEAR"
    """Linear transformation."""

    TRIANGULAR = "TRIANGULAR"
    """Transformation using triangular-shaped function."""

    BELL = "BELL"
    """Transformation using bell-shaped function."""
