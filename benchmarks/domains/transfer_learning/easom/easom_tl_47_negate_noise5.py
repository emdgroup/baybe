"""Easom function with transfer learning, negated function and a noise_std of 0.05."""

from __future__ import annotations

import pandas as pd

from benchmarks.definition import (
    ConvergenceBenchmark,
    ConvergenceBenchmarkSettings,
)
from benchmarks.domains.transfer_learning.easom.base import (
    easom,
    easom_tl_noise,
)


def easom_tl_47_negate_noise5(settings: ConvergenceBenchmarkSettings) -> pd.DataFrame:
    """Benchmark function for transfer learning with the Easom function.

    Key characteristics:
    • Compares two negated versions of Easom function:
      - Target: standard negated Easom
      - Source: negated Easom with added noise (noise_std=0.05)
    • Uses 47 points per dimension
    • Tests transfer learning with different source data percentages:
      - 1% of source data
      - 5% of source data
      - 10% of source data
      - 20% of source data

    Args:
        settings: Configuration settings for the convergence benchmark

    Returns:
        DataFrame containing benchmark results
    """
    negate = True
    functions = {
        "Target_Function": lambda x: easom(x, negate=negate),
        "Source_Function": lambda x: easom(x, noise_std=0.05, negate=negate),
    }
    return easom_tl_noise(
        settings=settings,
        functions=functions,
        points_per_dim=47,
        percentages=[0.01, 0.05, 0.1, 0.2],
        negate=negate,
    )


benchmark_config = ConvergenceBenchmarkSettings(
    batch_size=2,
    n_doe_iterations=30,
    n_mc_iterations=100,
)

easom_tl_47_negate_noise5_benchmark = ConvergenceBenchmark(
    function=easom_tl_47_negate_noise5,
    optimal_target_values={"Target": 0.9635009628660742},
    settings=benchmark_config,
)
