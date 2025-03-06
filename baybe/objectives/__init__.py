"""BayBE objectives."""

from baybe.objectives.desirability import DesirabilityObjective
from baybe.objectives.pareto import ParetoObjective
from baybe.objectives.single import SingleTargetObjective

__all__ = [
    "SingleTargetObjective",
    "DesirabilityObjective",
    "ParetoObjective",
]
