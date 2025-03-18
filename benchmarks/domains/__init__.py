"""Benchmark domains."""

from benchmarks.definition.base import Benchmark
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
from benchmarks.domains.transfer_learning.easom.easom_tl_71_noise5 import (
    easom_tl_71_noise5_benchmark,
)
from benchmarks.domains.transfer_learning.easom.easom_tl_100_negate_noise5 import (
    easom_tl_100_negate_noise5_benchmark,
)
from benchmarks.domains.transfer_learning.hartmann.hartmann_tl_3_20_15 import (
    hartmann_tl_3_20_15_benchmark,
)
from benchmarks.domains.transfer_learning.hartmann.hartmann_tl_3_30_negate_15 import (
    hartmann_tl_3_30_negate_15_benchmark,
)
from benchmarks.domains.transfer_learning.michalewicz.michalewicz_tl_continuous import (
    michalewicz_tl_noise_benchmark,
)

BENCHMARKS: list[Benchmark] = [
    synthetic_2C1D_1C_benchmark,
    arylhalides_ChlorTrifluour_IodMeth_benchmark,
    arylhalides_IodoPyr_ChloroPyr_benchmark,
    direct_arylation_tl_temperature_benchmark,
    easom_tl_71_noise5_benchmark,
    easom_tl_100_negate_noise5_benchmark,
    hartmann_tl_3_30_negate_15_benchmark,
    hartmann_tl_3_20_15_benchmark,
    michalewicz_tl_noise_benchmark,
]


__all__ = ["BENCHMARKS"]
