"""Negated Hartmann in 3 dimensions, 30 points per dimension and noise_std=0.15."""

from __future__ import annotations

import pandas as pd

from benchmarks.definition import (
    ConvergenceBenchmark,
    ConvergenceBenchmarkSettings,
)
from benchmarks.domains.transfer_learning.hartmann.base import (
    abstract_hartmann_tl_noise,
    hartmann,
)


def hartmann_tl_3_30_negate_15(settings: ConvergenceBenchmarkSettings) -> pd.DataFrame:
    """Actual benchmark function.

    Compares two versions of the Hartmann function, where one has a
    noise_std of 0.15.
    """
    negate = True
    functions = {
        "Target_Function": lambda x: hartmann(x, negate=negate),
        "Source_Function": lambda x: hartmann(x, negate=False, noise_std=0.15),
    }
    return abstract_hartmann_tl_noise(
        settings=settings,
        functions=functions,
        points_per_dim=30,
        percentages=[0, 0.01, 0.1, 0.2],
        negate=negate,
    )


benchmark_config = ConvergenceBenchmarkSettings(
    batch_size=2,
    n_doe_iterations=20,
    n_mc_iterations=40,
)

hartmann_tl_3_30_negate_15_benchmark = ConvergenceBenchmark(
    function=hartmann_tl_3_30_negate_15,
    optimal_target_values={"Target": 3.8531243067415284},
    settings=benchmark_config,
)
