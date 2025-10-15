"""Target-related enumerations."""

from enum import Enum

from baybe.targets._deprecated import TargetMode, TargetTransformation  # noqa: F401


class MatchMode(Enum):
    """Enum representing modes for inexact matching of real-valued numbers."""

    LE = "<="
    """Less or equal."""

    EQ = "="
    """Equal."""

    GE = ">="
    """Greater or equal."""
