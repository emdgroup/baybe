"""Benchmark domains."""

from benchmarks.definition.base import Benchmark
from benchmarks.domains.hartmann_tl_inverted_noise import (
    hartmann_tl_inverted_noise_benchmark,
)
from benchmarks.domains.michalewicz_tl_noise import michalewicz_tl_noise_benchmark
from benchmarks.domains.synthetic_2C1D_1C import synthetic_2C1D_1C_benchmark
from benchmarks.domains.transfer_learning.aryl_halides.ChlorTrifluour_IodMeth import (
    arylhalides_ChlorTrifluour_IodMeth_benchmark,
)
from benchmarks.domains.transfer_learning.aryl_halides.IodoPyr_ChloroPyr import (
    arylhalides_IodoPyr_ChloroPyr_benchmark,
)
from benchmarks.domains.transfer_learning.direct_arylation.temperature import (
    direct_arylation_tl_temperature_benchmark,
)
from benchmarks.domains.transfer_learning.easom.easom_tl_200_noise5 import (
    easom_tl_200_noise5_benchmark,
)
from benchmarks.domains.transfer_learning.easom.easom_tl_500_negate_noise5 import (
    easom_tl_500_negate_noise5_benchmark,
)

BENCHMARKS: list[Benchmark] = [
    synthetic_2C1D_1C_benchmark,
    arylhalides_ChlorTrifluour_IodMeth_benchmark,
    arylhalides_IodoPyr_ChloroPyr_benchmark,
    direct_arylation_tl_temperature_benchmark,
    easom_tl_200_noise5_benchmark,
    easom_tl_500_negate_noise5_benchmark,
    hartmann_tl_inverted_noise_benchmark,
    michalewicz_tl_noise_benchmark,
]


__all__ = ["BENCHMARKS"]
