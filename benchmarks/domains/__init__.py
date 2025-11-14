"""Benchmark domains."""

from benchmarks.definition.base import Benchmark
from benchmarks.domains.aryl_halides.convergence_tl import (
    aryl_halide_CT_I_BM_tl_benchmark,
    aryl_halide_CT_IM_tl_benchmark,
    aryl_halide_IP_CP_tl_benchmark,
)
from benchmarks.domains.aryl_halides.regression_tl import (
    aryl_halide_CT_I_BM_tl_regr_benchmark,
    aryl_halide_CT_IM_tl_regr_benchmark,
    aryl_halide_IP_CP_tl_regr_benchmark,
)
from benchmarks.domains.direct_arylation.convergence import (
    direct_arylation_multi_batch_benchmark,
    direct_arylation_single_batch_benchmark,
)
from benchmarks.domains.direct_arylation.convergence_tl import (
    direct_arylation_tl_temperature_benchmark,
)
from benchmarks.domains.direct_arylation.regression_tl import (
    direct_arylation_temperature_tl_regr_benchmark,
)
from benchmarks.domains.easom.convergence_tl import (
    easom_tl_47_negate_noise5_benchmark,
)
from benchmarks.domains.hartmann.convergence import (
    hartmann_3d_benchmark,
    hartmann_3d_discretized_benchmark,
    hartmann_6d_benchmark,
)
from benchmarks.domains.hartmann.convergence_tl import (
    hartmann_tl_3_20_15_benchmark,
)
from benchmarks.domains.michalewicz.convergence_tl import (
    michalewicz_tl_continuous_benchmark,
)
from benchmarks.domains.synthetic_2C1D_1C import synthetic_2C1D_1C_benchmark

BENCHMARKS: list[Benchmark] = [
    # Convergence Benchmarks
    direct_arylation_multi_batch_benchmark,
    direct_arylation_single_batch_benchmark,
    hartmann_3d_discretized_benchmark,
    synthetic_2C1D_1C_benchmark,
    hartmann_3d_benchmark,
    hartmann_6d_benchmark,
    # Transfer-Learning Convergence Benchmarks
    aryl_halide_CT_IM_tl_benchmark,
    aryl_halide_IP_CP_tl_benchmark,
    aryl_halide_CT_I_BM_tl_benchmark,
    direct_arylation_tl_temperature_benchmark,
    easom_tl_47_negate_noise5_benchmark,
    hartmann_tl_3_20_15_benchmark,
    michalewicz_tl_continuous_benchmark,
    # Transfer-Learning Regression Benchmarks
    direct_arylation_temperature_tl_regr_benchmark,
    aryl_halide_CT_I_BM_tl_regr_benchmark,
    aryl_halide_CT_IM_tl_regr_benchmark,
    aryl_halide_IP_CP_tl_regr_benchmark,
]


__all__ = ["BENCHMARKS"]
