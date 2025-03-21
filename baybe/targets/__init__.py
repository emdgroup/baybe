"""BayBE targets."""

from baybe.targets._deprecated import NumericalTarget
from baybe.targets.binary import BinaryTarget
from baybe.targets.enum import TargetMode, TargetTransformation

__all__ = [
    "BinaryTarget",
    "NumericalTarget",
    "TargetMode",
    "TargetTransformation",
]
