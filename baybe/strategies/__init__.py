"""BayBE strategies."""

from baybe.strategies.deprecation import Strategy
from baybe.strategies.scheduled import TwoPhaseStrategy

__all__ = [
    "TwoPhaseStrategy",
    "Strategy",
]
