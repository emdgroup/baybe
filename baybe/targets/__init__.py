"""BayBE targets."""

from baybe.targets.base import Target
from baybe.targets.deprecated import Objective
from baybe.targets.numerical import NumericalTarget

__all__ = [
    "Objective",
    "NumericalTarget",
    "Target",
]
