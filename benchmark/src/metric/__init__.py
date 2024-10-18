"""Metrics for evaluating the performance of algorithms."""

from src.metric.base import (
    GoalOrientedMetricInterface,
    Metric,
    NormalizationInterface,
)
from src.metric.error_metric import (
    NormalizedNegativeRootMeanSquaredErrorMetric,
)

__all__ = [
    "Metric",
    "NormalizationInterface",
    "GoalOrientedMetricInterface",
    "NormalizedNegativeRootMeanSquaredErrorMetric",
]
