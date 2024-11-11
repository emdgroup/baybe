"""This module contains the benchmarks for the different domains of the benchmark."""

from benchmark.definition import SingleExecutionBenchmark
from benchmark.domain.synthetic_3 import benchmark_synthetic_3

SINGLE_BENCHMARKS_TO_RUN: list[SingleExecutionBenchmark] = [
    benchmark_synthetic_3,
]

__all__ = ["SINGLE_BENCHMARKS_TO_RUN"]
