"""Metrics for evaluating the performance of algorithms."""

from benchmark.metric.base import (
    GoalOrientedMetricInterface,
    Metric,
    NormalizationInterface,
)
from benchmark.metric.error_metric import NormalizedNegativeRootMeanSquaredErrorMetric

__all__ = [
    "Metric",
    "NormalizationInterface",
    "GoalOrientedMetricInterface",
    "NormalizedNegativeRootMeanSquaredErrorMetric",
]
