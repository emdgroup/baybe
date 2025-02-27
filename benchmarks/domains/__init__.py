"""Benchmark domains."""

from benchmarks.definition.base import Benchmark
from benchmarks.domains.botorch_multitaskgp.arylhalides_tl_substance import (
    arylhalides_tl_substance_benchmark,
)
from benchmarks.domains.botorch_multitaskgp.direct_arylation_tl_temp import (
    direct_arylation_tl_temp_benchmark,
)
from benchmarks.domains.botorch_multitaskgp.easom_tl_noise import (
    easom_tl_noise_benchmark,
)
from benchmarks.domains.botorch_multitaskgp.hartmann_tl_inverted_noise import (
    hartmann_tl_inverted_noise_benchmark,
)
from benchmarks.domains.botorch_multitaskgp.michalewicz_tl_noise import (
    michalewicz_tl_noise_benchmark,
)

BENCHMARKS: list[Benchmark] = [
    arylhalides_tl_substance_benchmark,
    direct_arylation_tl_temp_benchmark,
    hartmann_tl_inverted_noise_benchmark,
    easom_tl_noise_benchmark,
    michalewicz_tl_noise_benchmark,
]

__all__ = ["BENCHMARKS"]
