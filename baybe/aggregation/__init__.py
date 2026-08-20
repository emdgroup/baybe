"""Aggregation functions for generality-oriented optimization."""

from baybe.aggregation.aggregations import (
    MeanAggregation,
    MinAggregation,
    SigmoidAggregation,
)
from baybe.aggregation.base import AggregationFunction

__all__ = [
    "AggregationFunction",
    "MeanAggregation",
    "MinAggregation",
    "SigmoidAggregation",
]
