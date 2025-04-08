"""Benchmark domains."""

from benchmarks.definition.base import Benchmark
from benchmarks.domains.synthetic_2C1D_1C import synthetic_2C1D_1C_benchmark
from benchmarks.domains.transfer_learning.aryl_halides.CT_IM_tl import (
    aryl_halide_CT_IM_tl_benchmark,
)
from benchmarks.domains.transfer_learning.aryl_halides.IP_CP_tl import (
    aryl_halide_IP_CP_tl_benchmark,
)
from benchmarks.domains.transfer_learning.direct_arylation.temperature_tl import (
    direct_arylation_tl_temperature_benchmark,
)
from benchmarks.domains.transfer_learning.easom.easom_tl_47_negate_noise5 import (
    easom_tl_47_negate_noise5_benchmark,
)
from benchmarks.domains.transfer_learning.easom.easom_tl_71_noise5 import (
    easom_tl_71_noise5_benchmark,
)
from benchmarks.domains.transfer_learning.hartmann.hartmann_tl_3_15_15 import (
    hartmann_tl_3_15_15_benchmark,
)
from benchmarks.domains.transfer_learning.hartmann.hartmann_tl_3_30_negate_15 import (
    hartmann_tl_3_30_negate_15_benchmark,
)
from benchmarks.domains.transfer_learning.michalewicz.michalewicz_tl_continuous import (
    michalewicz_tl_continuous_benchmark,
)

BENCHMARKS: list[Benchmark] = [
    synthetic_2C1D_1C_benchmark,
    aryl_halide_CT_IM_tl_benchmark,
    aryl_halide_IP_CP_tl_benchmark,
    direct_arylation_tl_temperature_benchmark,
    easom_tl_71_noise5_benchmark,
    easom_tl_47_negate_noise5_benchmark,
    hartmann_tl_3_30_negate_15_benchmark,
    hartmann_tl_3_15_15_benchmark,
    michalewicz_tl_continuous_benchmark,
]


__all__ = ["BENCHMARKS"]
