"""Benchmark domains."""

from benchmarks.definition.base import Benchmark

# from benchmarks.domains.kernel_presets.easom_tl_noise import \
# easom_tl_noise_benchmark
from benchmarks.domains.kernel_presets.hartmann_tl_inverted_noise import (
    hartmann_tl_inverted_noise_benchmark,
)

BENCHMARKS: list[Benchmark] = [
    hartmann_tl_inverted_noise_benchmark,
]

__all__ = ["BENCHMARKS"]
