"""Discretized version of the Hartmann function in 3 dimensions as a benchmark."""

from __future__ import annotations

import numpy as np
from botorch.test_functions.synthetic import Hartmann
from pandas import DataFrame

from baybe.campaign import Campaign
from baybe.parameters import NumericalDiscreteParameter
from baybe.recommenders import RandomRecommender
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget
from baybe.utils.dataframe import arrays_to_dataframes
from benchmarks.definition.convergence import (
    ConvergenceBenchmark,
    ConvergenceBenchmarkSettings,
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

    target = NumericalTarget(name="target", minimmize=True)
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

    test_function = Hartmann(dim=3)

    lookup_discretized = arrays_to_dataframes(
        [p.name for p in parameters], [target.name], use_torch=True
    )(test_function)

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
