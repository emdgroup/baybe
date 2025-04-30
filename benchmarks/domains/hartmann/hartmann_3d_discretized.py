"""Discretized version of the Hartmann function in 3 dimensions as a benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame

from baybe.campaign import Campaign
from baybe.parameters import NumericalDiscreteParameter
from baybe.recommenders import RandomRecommender
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget
from benchmarks.definition.convergence import (
    ConvergenceBenchmark,
    ConvergenceBenchmarkSettings,
)
from benchmarks.domains.hartmann.hartmann_3d import _hartmann_3d


def lookup_discretized(df: pd.DataFrame, /) -> pd.DataFrame:
    """Dataframe-based lookup callable for the discretized version.

    Args:
        df: DataFrame containing the discrete parameter values.

    Returns:
        DataFrame with calculated target values.
    """
    # Convert discrete values to continuous by dividing by the number of discrete steps
    x1_cont = df["x1"].to_numpy()
    x2_cont = df["x2"].to_numpy()
    x3_cont = df["x3"].to_numpy()

    # Stack into array for the hartmann function
    points = np.column_stack((x1_cont, x2_cont, x3_cont))

    return pd.DataFrame(
        _hartmann_3d(points),
        columns=["target"],
        index=df.index,
    )


def hartmann_3d_discretized(settings: ConvergenceBenchmarkSettings) -> DataFrame:
    """Benchmark function with the discretized Hartmann 3D test function.

    Key characteristics:
    • Parameters:
      - x1: Discretisation of [0,1] with 25 steps
      - x2: Discretisation of [0,1] with 25 steps
      - x3: Discretisation of [0,1] with 25 steps
    • Output: Continuous
    • Objective: Minimization
    • Tests multiple recommenders:
      - Random Recommender
      - Default Recommender

    Args:
        settings: Configuration settings for the convergence benchmark.

    Returns:
        DataFrame containing benchmark results.
    """
    parameters = [
        NumericalDiscreteParameter("x1", np.linspace(0, 1, 25)),
        NumericalDiscreteParameter("x2", np.linspace(0, 1, 25)),
        NumericalDiscreteParameter("x3", np.linspace(0, 1, 25)),
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
        lookup_discretized,
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

hartmann_3d_discretized_benchmark = ConvergenceBenchmark(
    function=hartmann_3d_discretized,
    optimal_target_values={"target": -3.86278},
    settings=benchmark_config,
)
