"""Hartmann convergence benchmark.

Scenario 1:
    Discretized version of the Hartmann function in 3 dimensions.

Scenario 2:
    Continuous version of the Hartmann function in 3 dimensions.

Scenario 3:
    Continuous version of the Hartmann function in 6 dimensions.
"""

from __future__ import annotations

import numpy as np
from botorch.test_functions.synthetic import Hartmann
from pandas import DataFrame

from baybe.campaign import Campaign
from baybe.parameters import NumericalContinuousParameter, NumericalDiscreteParameter
from baybe.recommenders import RandomRecommender
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget
from baybe.utils.dataframe import arrays_to_dataframes
from benchmarks.definition.base import RunMode
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

    target = NumericalTarget(name="target", minimize=True)
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
        random_seed=settings.random_seed,
    )


def hartmann_3d(settings: ConvergenceBenchmarkSettings) -> DataFrame:
    """Benchmark function with the Hartmann 3D test function.

    Key characteristics:
    • Parameters:
      - x1: Continuous [0, 1]
      - x2: Continuous [0, 1]
      - x3: Continuous [0, 1]
    • Output: Continuous
    • Objective: Minimization
    • Optimal inputs: {x1: 0.114614, x2: 0.555649, x3: 0.852547}
    • Optimal output: -3.86278
    • Tests multiple recommenders:
      - Random Recommender
      - Default Recommender

    Args:
        settings: Configuration settings for the convergence benchmark.

    Returns:
        DataFrame containing benchmark results.
    """
    parameters = [
        NumericalContinuousParameter("x1", (0.0, 1.0)),
        NumericalContinuousParameter("x2", (0.0, 1.0)),
        NumericalContinuousParameter("x3", (0.0, 1.0)),
    ]

    target = NumericalTarget(name="target", minimize=True)
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

    hartmann = Hartmann(dim=3)

    lookup = arrays_to_dataframes(
        [p.name for p in parameters], [target.name], use_torch=True
    )(hartmann)

    return simulate_scenarios(
        scenarios,
        lookup,
        batch_size=settings.batch_size,
        n_doe_iterations=settings.n_doe_iterations,
        n_mc_iterations=settings.n_mc_iterations,
        impute_mode="error",
        random_seed=settings.random_seed,
    )


def hartmann_6d(settings: ConvergenceBenchmarkSettings) -> DataFrame:
    """Benchmark function with the Hartmann 6D test function.

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

    target = NumericalTarget(name="target", minimize=True)
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

    hartmann = Hartmann(dim=6)

    lookup = arrays_to_dataframes(
        [p.name for p in parameters], [target.name], use_torch=True
    )(hartmann)

    return simulate_scenarios(
        scenarios,
        lookup,
        batch_size=settings.batch_size,
        n_doe_iterations=settings.n_doe_iterations,
        n_mc_iterations=settings.n_mc_iterations,
        impute_mode="error",
        random_seed=settings.random_seed,
    )


benchmark_config = ConvergenceBenchmarkSettings(
    batch_size_settings={
        RunMode.DEFAULT: 5,
        RunMode.SMOKETEST: 2,
    },
    n_doe_iterations_settings={
        RunMode.DEFAULT: 30,
        RunMode.SMOKETEST: 2,
    },
    n_mc_iterations_settings={
        RunMode.DEFAULT: 100,
        RunMode.SMOKETEST: 2,
    },
)

hartmann_3d_discretized_benchmark = ConvergenceBenchmark(
    function=hartmann_3d_discretized,
    optimal_target_values={"target": -3.86278},
    settings=benchmark_config,
)

hartmann_3d_benchmark = ConvergenceBenchmark(
    function=hartmann_3d,
    optimal_target_values={"target": -3.86278},
    settings=benchmark_config,
)

hartmann_6d_benchmark = ConvergenceBenchmark(
    function=hartmann_6d,
    optimal_target_values={"target": -3.86278},
    settings=benchmark_config,
)
