"""Hartmann function in 3 dimensions, 50 points per dimension and noise_std=0.15."""

from __future__ import annotations

import pandas as pd

from benchmarks.definition import (
    ConvergenceBenchmark,
    ConvergenceBenchmarkSettings,
)
from benchmarks.domains.transfer_learning.hartmann.base import (
    hartmann,
    hartmann_tl_noise,
)


def hartmann_tl_3_15_15(settings: ConvergenceBenchmarkSettings) -> pd.DataFrame:
    """Benchmark function for transfer learning with the Hartmann function in 3D.

    Key characteristics:
    • Compares two versions of Hartmann function:
      - Target: standard Hartmann
      - Source: Hartmann with added noise (noise_std=0.15)
    • Uses 15 points per dimension
    • Tests transfer learning with different source data percentages:
      - 1% of source data
      - 10% of source data
      - 20% of source data

    Args:
        settings: Configuration settings for the convergence benchmark.

    Returns:
        DataFrame containing benchmark results.
    """
    negate = False
    functions = {
        "Target_Function": lambda x: hartmann(x, negate=negate),
        "Source_Function": lambda x: hartmann(x, negate=negate, noise_std=0.15),
    }
    return hartmann_tl_noise(
        settings=settings,
        functions=functions,
        points_per_dim=15,
        percentages=[0.01, 0.1, 0.2],
        negate=negate,
    )


benchmark_config = ConvergenceBenchmarkSettings(
    batch_size=2,
    n_doe_iterations=25,
    n_mc_iterations=70,
)

hartmann_tl_3_15_15_benchmark = ConvergenceBenchmark(
    function=hartmann_tl_3_15_15,
    optimal_target_values={"Target": -3.851831124860353},
    settings=benchmark_config,
)
