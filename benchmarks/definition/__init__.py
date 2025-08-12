"""Benchmark definitions."""

from benchmarks.definition.base import Benchmark, BenchmarkSettings, RunMode
from benchmarks.definition.convergence import (
    ConvergenceBenchmark,
    ConvergenceBenchmarkSettings,
)

__all__ = [
    "Benchmark",
    "BenchmarkSettings",
    "ConvergenceBenchmark",
    "ConvergenceBenchmarkSettings",
    "RunMode",
]
