"""Benchmark domains."""

from benchmarks.definition import Benchmark
from benchmarks.domains.synthetic_3 import benchmark_synthetic_3

BENCHMARKS: list[Benchmark] = [
    benchmark_synthetic_3,
]

__all__ = ["BENCHMARKS"]
