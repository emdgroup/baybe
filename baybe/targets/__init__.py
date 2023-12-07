"""BayBE targets."""

from baybe.targets.deprecation import Objective
from baybe.targets.enum import TargetMode
from baybe.targets.numerical import NumericalTarget

__all__ = [
    "NumericalTarget",
    "Objective",
    "TargetMode",
]
