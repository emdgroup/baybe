"""Metrics for evaluating the performance of algorithms."""

from benchmark.metric.base import Metric, NormalizationInterface
from benchmark.metric.simple_regret import NormalizedSimpleRegret

__all__ = ["Metric", "NormalizationInterface", "NormalizedSimpleRegret"]
