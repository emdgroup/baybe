"""Aryl halide benchmark for transfer learning.

Source task: 2-iodopyridine
Target task: 3-chloropyridine
"""

from __future__ import annotations

import pandas as pd

from baybe.utils.random import temporary_seed
from benchmarks.definition import (
    ConvergenceBenchmark,
    ConvergenceBenchmarkSettings,
)
from benchmarks.domains.transfer_learning.aryl_halides.base import (
    aryl_halide_tl_substance_benchmark,
)


def aryl_halide_IP_CP_tl(
    settings: ConvergenceBenchmarkSettings,
) -> pd.DataFrame:
    """Benchmark function for transfer learning with aryl halide reactions.

    Key characteristics:
    • Compares transfer learning vs. non-transfer learning approaches
    • Source task: 2-iodopyridine
    • Target task: 3-chloropyridine
    • Tests varying amounts of source data:
      - 1% of source data
      - 5% of source data
      - 10% of source data
      - 20% of source data
    • Optimal parameters:
      - Base: "P2Et"
      - Ligand: "t-BuXPhos"
      - Additive: "4-phenylisoxazole"
    • Optimal yield: 68.76%

    Args:
        settings: Configuration settings for the convergence benchmark

    Returns:
        DataFrame containing benchmark results
    """
    with temporary_seed(settings.random_seed):
        return aryl_halide_tl_substance_benchmark(
            settings=settings,
            source_tasks=["2-iodopyridine"],
            target_tasks=["3-chloropyridine"],
            percentages=[0.01, 0.05, 0.1, 0.2],
        )


benchmark_config = ConvergenceBenchmarkSettings(
    batch_size=2,
    n_doe_iterations=25,
    n_mc_iterations=60,
)

aryl_halide_IP_CP_tl_benchmark = ConvergenceBenchmark(
    function=aryl_halide_IP_CP_tl,
    optimal_target_values={"yield": 68.76495224},
    settings=benchmark_config,
)
