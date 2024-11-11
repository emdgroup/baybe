"""Benchmark domains."""

from benchmarks.definition import Benchmark
from benchmarks.domains import synthetic_2C1D_1C

BENCHMARKS: list[Benchmark] = [
    synthetic_2C1D_1C.benchmark,
]

__all__ = ["BENCHMARKS"]
