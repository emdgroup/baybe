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
from benchmarks.domains.transfer_learning.aryl_halides.base import (
    aryl_halide_tl_substance_benchmark,
)


def aryl_halide_IP_CP_tl(
    settings: ConvergenceBenchmarkSettings,
) -> pd.DataFrame:
    """Actual benchmark function.

    Optimal Inputs:
        base:       "P2Et",
        ligand:     "t-BuXPhos",
        additive:   "4-phenylisoxazole"
    Optimal Output: 68.76495224
    """
    return aryl_halide_tl_substance_benchmark(
        settings=settings,
        source_tasks=["2-iodopyridine"],
        target_tasks=["3-chloropyridine"],
        percentages=[0.01, 0.1, 0.2],
    )


benchmark_config = ConvergenceBenchmarkSettings(
    batch_size=2,
    n_doe_iterations=30,
    n_mc_iterations=100,
)

aryl_halide_IP_CP_tl_benchmark = ConvergenceBenchmark(
    function=aryl_halide_IP_CP_tl,
    optimal_target_values={"yield": 68.76495224},
    settings=benchmark_config,
)
