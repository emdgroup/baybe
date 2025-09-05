"""Aryl halide benchmark for transfer learning.

Source task: 1-chloro-4-(trifluoromethyl)benzene
Target task: 1-iodo-4-methoxybenzene
"""

from __future__ import annotations

import pandas as pd

from benchmarks.definition import (
    ConvergenceBenchmark,
    ConvergenceBenchmarkSettings,
)
from benchmarks.definition.base import RunMode
from benchmarks.domains.transfer_learning.aryl_halides.base import (
    aryl_halide_tl_substance_benchmark,
)


def aryl_halide_CT_IM_tl(
    settings: ConvergenceBenchmarkSettings,
) -> pd.DataFrame:
    """Benchmark function for transfer learning with aryl halide reactions.

    Key characteristics:
    • Compares transfer learning vs. non-transfer learning approaches
    • Source task: 1-chloro-4-(trifluoromethyl)benzene
    • Target task: 1-iodo-4-methoxybenzene
    • Tests varying amounts of source data:
      - 1% of source data
      - 5% of source data
      - 10% of source data
      - 20% of source data
    • Optimal parameters:
      - Base: "MTBD"
      - Ligand: "AdBrettPhos"
      - Additive: "N,N-dibenzylisoxazol-3-amine"
    • Optimal yield: 68.25%

    Args:
        settings: Configuration settings for the convergence benchmark

    Returns:
        DataFrame containing benchmark results
    """
    return aryl_halide_tl_substance_benchmark(
        settings=settings,
        source_tasks=["1-chloro-4-(trifluoromethyl)benzene"],
        target_tasks=["1-iodo-4-methoxybenzene"],
        percentages=[0.01, 0.05, 0.1, 0.2],
    )


benchmark_config = ConvergenceBenchmarkSettings(
    batch_size_settings={
        RunMode.DEFAULT: 2,
        RunMode.SMOKETEST: 2,
    },
    n_doe_iterations_settings={
        RunMode.DEFAULT: 25,
        RunMode.SMOKETEST: 2,
    },
    n_mc_iterations_settings={
        RunMode.DEFAULT: 50,
        RunMode.SMOKETEST: 2,
    },
)

aryl_halide_CT_IM_tl_benchmark = ConvergenceBenchmark(
    function=aryl_halide_CT_IM_tl,
    optimal_target_values={"yield": 68.24812709999999},
    settings=benchmark_config,
)
