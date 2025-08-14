"""Hartmann function in 6 dimensions as a benchmark."""

from __future__ import annotations

from botorch.test_functions.synthetic import Hartmann
from pandas import DataFrame

from baybe.campaign import Campaign
from baybe.parameters import NumericalContinuousParameter
from baybe.recommenders import RandomRecommender
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget
from baybe.utils.dataframe import arrays_to_dataframes
from benchmarks.definition.convergence import (
    ConvergenceBenchmark,
    ConvergenceBenchmarkSettings,
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
