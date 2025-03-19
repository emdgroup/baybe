"""Aryl-Halide benchmark for transfer learning.

Source task: 1-chloro-4-(trifluoromethyl)benzene
Target task: 1-iodo-4-methoxybenzene
"""

from __future__ import annotations

import pandas as pd

from benchmarks.definition import (
    ConvergenceBenchmark,
    ConvergenceBenchmarkSettings,
)
from benchmarks.domains.transfer_learning.aryl_halides.base import (
    abstract_aryl_halide_tl_substance_benchmark,
)


def aryl_halide_ChlorTrifluour_IodoMeth(
    settings: ConvergenceBenchmarkSettings,
) -> pd.DataFrame:
    """Actual benchmark function.

    Optimal Inputs:
        base:       "MTBD",
        ligand:     "AdBrettPhos",
        additive:   "N,N-dibenzylisoxazol-3-amine"
    Optimal Output: 68.24812709999999
    """
    return abstract_aryl_halide_tl_substance_benchmark(
        settings=settings,
        source_tasks=["1-chloro-4-(trifluoromethyl)benzene"],
        target_tasks=["1-iodo-4-methoxybenzene"],
        percentages=[0.01, 0.05, 0.1, 0.2],
    )


benchmark_config = ConvergenceBenchmarkSettings(
    batch_size=2,
    n_doe_iterations=20,
    n_mc_iterations=50,
)

aryl_halide_ChlorTrifluour_IodMeth_benchmark = ConvergenceBenchmark(
    function=aryl_halide_ChlorTrifluour_IodoMeth,
    optimal_target_values={"yield": 68.24812709999999},
    settings=benchmark_config,
)
