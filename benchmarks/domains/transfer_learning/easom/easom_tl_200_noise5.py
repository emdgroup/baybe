"""Easom function with transfer learning, negated function and a noise_std of 0.05."""

from __future__ import annotations

import pandas as pd

from benchmarks.definition import (
    ConvergenceBenchmark,
    ConvergenceBenchmarkSettings,
)
from benchmarks.domains.transfer_learning.easom.base import (
    abstract_easom_tl_noise,
    easom,
)


def easom_tl_200_noise5(settings: ConvergenceBenchmarkSettings) -> pd.DataFrame:
    """Actual benchmark function.

    Compares two versions of the Easom function, where one has a
    noise_std of 0.05.
    """
    negate = False
    functions = {
        "Target_Function": lambda x: easom(x, negate=negate),
        "Source_Function": lambda x: easom(x, noise_std=0.05, negate=negate),
    }
    return abstract_easom_tl_noise(
        settings=settings,
        functions=functions,
        points_per_dim=200,
        percentages=[0.01, 0.02, 0.05, 0.1, 0.2],
        negate=negate,
    )


benchmark_config = ConvergenceBenchmarkSettings(
    batch_size=2,
    n_doe_iterations=10,
    n_mc_iterations=50,
)

easom_tl_200_noise5_benchmark = ConvergenceBenchmark(
    function=easom_tl_200_noise5,
    optimal_target_values={"Target": -0.652085},
    settings=benchmark_config,
)
