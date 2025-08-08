"""Benchmark definitions."""

from benchmarks.definition.base import (
    Benchmark,
    BenchmarkSettings,
)
from benchmarks.definition.convergence import (
    ConvergenceBenchmark,
    ConvergenceBenchmarkSettings,
)
from benchmarks.definition.regression import (
    TransferLearningRegression,
    TransferLearningRegressionSettings,
)

__all__ = [
    "Benchmark",
    "BenchmarkSettings",
    "ConvergenceBenchmark",
    "ConvergenceBenchmarkSettings",
    "TransferLearningRegression",
    "TransferLearningRegressionSettings",
]
