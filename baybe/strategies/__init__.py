"""BayBE strategies."""

from baybe.strategies.deprecation import Strategy
from baybe.strategies.scheduled import SplitStrategy

__all__ = [
    "SplitStrategy",
    "Strategy",
]
