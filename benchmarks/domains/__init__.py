"""Benchmark domains."""

from benchmarks.definition import BenchmarkDefinition
from benchmarks.domains import synthetic_2C1D_1C

BENCHMARKS: list[BenchmarkDefinition] = [
    synthetic_2C1D_1C.benchmark,
]

__all__ = ["BENCHMARKS"]
