"""Benchmark domains."""

from benchmarks.definition.base import Benchmark
from benchmarks.domains.arylhalides_tl_substance import (
    arylhalides_tl_substance_benchmark,
)
from benchmarks.domains.direct_arylation_tl_temp import (
    direct_arylation_tl_temp_benchmark,
)
from benchmarks.domains.easom_tl_noise import easom_tl_noise_benchmark
from benchmarks.domains.hartmann_tl_inverted_noise import (
    hartmann_tl_inverted_noise_benchmark,
)
from benchmarks.domains.michalewicz_tl_noise import michalewicz_tl_noise_benchmark

# from benchmarks.domains.synthetic_2C1D_1C import synthetic_2C1D_1C_benchmark

BENCHMARKS: list[Benchmark] = [
    # synthetic_2C1D_1C_benchmark,
    arylhalides_tl_substance_benchmark,
    direct_arylation_tl_temp_benchmark,
    hartmann_tl_inverted_noise_benchmark,
    easom_tl_noise_benchmark,
    michalewicz_tl_noise_benchmark,
]


__all__ = ["BENCHMARKS"]
