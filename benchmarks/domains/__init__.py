"""Benchmark domains."""

from benchmarks.definition.base import Benchmark
from benchmarks.domains.direct_arylation.direct_arylation_multi_batch import (
    direct_arylation_multi_batch_benchmark,
)
from benchmarks.domains.direct_arylation.direct_arylation_single_batch import (
    direct_arylation_single_batch_benchmark,
)
from benchmarks.domains.hartmann.hartmann_3d import hartmann_3d_benchmark
from benchmarks.domains.hartmann.hartmann_3d_discretized import (
    hartmann_3d_discretized_benchmark,
)
from benchmarks.domains.hartmann.hartmann_6d import hartmann_6d_benchmark
from benchmarks.domains.synthetic_2C1D_1C import synthetic_2C1D_1C_benchmark
from benchmarks.domains.transfer_learning.aryl_halides.sou_CT_I_tar_BM import (
    aryl_halide_CT_I_BM_tl_benchmark,
)
from benchmarks.domains.transfer_learning.aryl_halides.sou_CT_tar_IM import (
    aryl_halide_CT_IM_tl_benchmark,
)
from benchmarks.domains.transfer_learning.aryl_halides.sout_IP_tar_CP import (
    aryl_halide_IP_CP_tl_benchmark,
)
from benchmarks.domains.transfer_learning.direct_arylation.temperature_tl import (
    direct_arylation_tl_temperature_benchmark,
)
from benchmarks.domains.transfer_learning.easom.easom_tl_47_negate_noise5 import (
    easom_tl_47_negate_noise5_benchmark,
)
from benchmarks.domains.transfer_learning.hartmann.hartmann_tl_3_20_15 import (
    hartmann_tl_3_20_15_benchmark,
)
from benchmarks.domains.transfer_learning.michalewicz.michalewicz_tl_continuous import (
    michalewicz_tl_continuous_benchmark,
)

BENCHMARKS: list[Benchmark] = [
    direct_arylation_multi_batch_benchmark,
    direct_arylation_single_batch_benchmark,
    hartmann_3d_discretized_benchmark,
    synthetic_2C1D_1C_benchmark,
    hartmann_3d_benchmark,
    hartmann_6d_benchmark,
    aryl_halide_CT_IM_tl_benchmark,
    aryl_halide_IP_CP_tl_benchmark,
    aryl_halide_CT_I_BM_tl_benchmark,
    direct_arylation_tl_temperature_benchmark,
    easom_tl_47_negate_noise5_benchmark,
    hartmann_tl_3_20_15_benchmark,
    michalewicz_tl_continuous_benchmark,
]


__all__ = ["BENCHMARKS"]
