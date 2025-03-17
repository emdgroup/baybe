"""Hartmann function in 3 dimensions, 50 points per dimension and noise_std=0.15."""

from __future__ import annotations

import pandas as pd
import torch
from botorch.test_functions.synthetic import Hartmann

from benchmarks.definition import (
    ConvergenceBenchmark,
    ConvergenceBenchmarkSettings,
)
from benchmarks.domains.transfer_learning.hartmann.base import (
    abstract_hartmann_tl_noise,
)


def hartmann_tl_3_50_15(settings: ConvergenceBenchmarkSettings) -> pd.DataFrame:
    """Actual benchmark function.

    Compares two versions of the Hartmann function, where one has a
    noise_std of 0.15.
    """
    negate = False
    dim = 3
    functions = {
        "Target_Function": lambda x: Hartmann(dim=dim, negate=negate)
        .forward(torch.tensor(x))
        .item(),
        "Source_Function": lambda x: Hartmann(dim=dim, negate=False, noise_std=0.15)
        .forward(torch.tensor(x))
        .item(),
    }
    return abstract_hartmann_tl_noise(
        settings=settings,
        functions=functions,
        points_per_dim=50,
        dim=dim,
        percentages=[0.01, 0.02, 0.05, 0.1, 0.2],
        negate=negate,
    )


benchmark_config = ConvergenceBenchmarkSettings(
    batch_size=2,
    n_doe_iterations=10,
    n_mc_iterations=50,
)

hartmann_tl_3_50_15_benchmark = ConvergenceBenchmark(
    function=hartmann_tl_3_50_15,
    optimal_target_values={"Target": -3.8598023524495533},
    settings=benchmark_config,
)
