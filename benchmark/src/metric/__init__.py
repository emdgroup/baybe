"""Metrics for evaluating the performance of algorithms."""

from benchmark.src.metric.auc import NormalizedAreaUnderTheCurve
from benchmark.src.metric.base import (
    Metric,
    NormalizationInterface,
)

__all__ = [
    "Metric",
    "NormalizationInterface",
    "NormalizedAreaUnderTheCurve",
]
