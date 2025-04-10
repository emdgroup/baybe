"""Hartmann function in 6 dimensions as a benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame

from baybe.campaign import Campaign
from baybe.parameters import NumericalContinuousParameter
from baybe.recommenders import RandomRecommender
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget
from benchmarks.definition.convergence import (
    ConvergenceBenchmark,
    ConvergenceBenchmarkSettings,
)


def _hartmann_6d(x: np.ndarray) -> np.ndarray:
    """Calculate the Hartmann function in 6D.

    Args:
        x: Input array of shape (n, 6) where n is the number of points

    Returns:
        Array of function values
    """
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array(
        [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ]
    )
    P = 1e-4 * np.array(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381.0],
        ]
    )

    outer = 0
    for i in range(4):
        inner = 0
        for j in range(6):
            inner += A[i, j] * ((x[:, j] - P[i, j]) ** 2)
        outer += alpha[i] * np.exp(-inner)

    return -outer


def lookup(df: pd.DataFrame, /) -> pd.DataFrame:
    """Dataframe-based lookup callable used as the loop-closing element."""
    return pd.DataFrame(
        _hartmann_6d(df[["x1", "x2", "x3", "x4", "x5", "x6"]].to_numpy()),
        columns=["target"],
        index=df.index,
    )


def hartmann_6d(settings: ConvergenceBenchmarkSettings) -> DataFrame:
    """Benchmark function with the Hartmann 3D test function.

    Key characteristics:
    • Parameters:
      - x1: Continuous [0, 1]
      - x2: Continuous [0, 1]
      - x3: Continuous [0, 1]
      - x4: Continuous [0, 1]
      - x5: Continuous [0, 1]
      - x6: Continuous [0, 1]
    • Output: Continuous
    • Objective: Minimization
    • Optimal inputs:
        {x1: 0.20169, x2: 0.150011, x3: 0.476874,
        x4: 0.275332, x5: 0.311652, x6: 0.6573}
    • Optimal output: -3.86278
    • Tests multiple recommenders:
      - Random Recommender
      - Default Recommender

    Args:
        settings: Configuration settings for the convergence benchmark

    Returns:
        DataFrame containing benchmark results
    """
    parameters = [
        NumericalContinuousParameter("x1", (0.0, 1.0)),
        NumericalContinuousParameter("x2", (0.0, 1.0)),
        NumericalContinuousParameter("x3", (0.0, 1.0)),
        NumericalContinuousParameter("x4", (0.0, 1.0)),
        NumericalContinuousParameter("x5", (0.0, 1.0)),
        NumericalContinuousParameter("x6", (0.0, 1.0)),
    ]

    target = NumericalTarget(name="target", mode="MIN")
    searchspace = SearchSpace.from_product(parameters=parameters)
    objective = target.to_objective()

    scenarios: dict[str, Campaign] = {
        "Random Recommender": Campaign(
            searchspace=searchspace,
            recommender=RandomRecommender(),
            objective=objective,
        ),
        "Default Recommender": Campaign(
            searchspace=searchspace,
            objective=objective,
        ),
    }

    return simulate_scenarios(
        scenarios,
        lookup,
        batch_size=settings.batch_size,
        n_doe_iterations=settings.n_doe_iterations,
        n_mc_iterations=settings.n_mc_iterations,
        impute_mode="error",
    )


benchmark_config = ConvergenceBenchmarkSettings(
    batch_size=5,
    n_doe_iterations=30,
    n_mc_iterations=100,
)

hartmann_6d_benchmark = ConvergenceBenchmark(
    function=hartmann_6d,
    optimal_target_values={"target": -3.86278},
    settings=benchmark_config,
)
