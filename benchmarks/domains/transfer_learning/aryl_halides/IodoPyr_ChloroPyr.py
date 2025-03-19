"""Aryl-Halide benchmark for transfer learning.

As source parameter, this benchmark uses 2-iodopyridine.
As target parameter, this benchmark uses 3-chloropyridine.
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


def aryl_halide_IodoPyr_ChloroPyr(
    settings: ConvergenceBenchmarkSettings,
) -> pd.DataFrame:
    """Actual benchmark function.

    Optimal Inputs:
        base:       "P2Et",
        ligand:     "t-BuXPhos",
        additive:   "4-phenylisoxazole"
    Optimal Output: 68.76495224
    """
    return abstract_aryl_halide_tl_substance_benchmark(
        settings=settings,
        source_tasks=["2-iodopyridine"],
        target_tasks=["3-chloropyridine"],
        percentages=[0.01, 0.05, 0.1, 0.2],
    )


benchmark_config = ConvergenceBenchmarkSettings(
    batch_size=2,
    n_doe_iterations=17,
    n_mc_iterations=50,
)

aryl_halide_IodoPyr_ChloroPyr_benchmark = ConvergenceBenchmark(
    function=aryl_halide_IodoPyr_ChloroPyr,
    optimal_target_values={"yield": 68.76495224},
    settings=benchmark_config,
)
