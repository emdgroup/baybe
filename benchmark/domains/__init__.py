"""This module contains the benchmarks for the different domains of the benchmark."""

from src import SingleExecutionBenchmark

from domains.direct_arylation import benchmark_direct_arylation
from domains.synthetic_1 import benchmark_synthetic_1

SINGE_BENCHMARKS_TO_RUN: list[SingleExecutionBenchmark] = [
    benchmark_synthetic_1,
    benchmark_direct_arylation,
]

__all__ = ["SINGE_BENCHMARKS_TO_RUN"]
