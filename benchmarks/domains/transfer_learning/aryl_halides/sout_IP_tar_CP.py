"""Aryl halide benchmark for transfer learning.

Source task: 2-iodopyridine
Target task: 3-chloropyridine
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
    return aryl_halide_tl_substance_benchmark(
        settings=settings,
        source_tasks=["2-iodopyridine"],
        target_tasks=["3-chloropyridine"],
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
        RunMode.DEFAULT: 60,
        RunMode.SMOKETEST: 2,
    },
)

aryl_halide_IP_CP_tl_benchmark = ConvergenceBenchmark(
    function=aryl_halide_IP_CP_tl,
    optimal_target_values={"yield": 68.76495224},
    settings=benchmark_config,
)
