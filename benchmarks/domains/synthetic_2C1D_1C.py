"""Synthetic function with two continuous and one discrete input."""

from uuid import UUID

import numpy as np
from numpy import pi, sin, sqrt
from pandas import DataFrame

from baybe.campaign import Campaign
from baybe.parameters import NumericalContinuousParameter, NumericalDiscreteParameter
from baybe.recommenders.pure.nonpredictive.sampling import RandomRecommender
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget, TargetMode
from benchmarks.definition import Benchmark
from benchmarks.definition.config import (
    DEFAULT_RECOMMENDER,
    RecommenderConvergenceAnalysis,
)


def lookup(z: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Lookup function.

    Inputs:
        z   discrete   {1,2,3,4}
        x   continuous [-2*pi, 2*pi]
        y   continuous [-2*pi, 2*pi]
    Output: continuous
    Best value: 4.09685
    """
    try:
        assert np.all(-2 * pi <= x) and np.all(x <= 2 * pi)
        assert np.all(-2 * pi <= y) and np.all(y <= 2 * pi)
        assert np.all(np.isin(z, [1, 2, 3, 4]))
    except AssertionError:
        raise ValueError("Inputs are not in the valid ranges.")

    return (
        (z == 1) * sin(x) * (1 + sin(y))
        + (z == 2) * (x * sin(0.9 * x) + sin(x) * sin(y))
        + (z == 3) * (sqrt(x + 8) * sin(x) + sin(x) * sin(y))
        + (z == 4) * (x * sin(1.666 * sqrt(x + 8)) + sin(x) * sin(y))
    )


def benchmark_callable(scenario_config: RecommenderConvergenceAnalysis) -> DataFrame:
    """Run different recommenders against the test function."""
    parameters = [
        NumericalContinuousParameter("x", (-2 * pi, 2 * pi)),
        NumericalContinuousParameter("y", (-2 * pi, 2 * pi)),
        NumericalDiscreteParameter("z", (1, 2, 3, 4)),
    ]

    objective = NumericalTarget(name="target", mode=TargetMode.MAX).to_objective()

    scenarios: dict[str, Campaign] = {}
    for scenario_name, recommender in scenario_config.recommenders.items():
        recommender_dct = (
            {} if recommender is DEFAULT_RECOMMENDER else {"recommender": recommender}
        )
        campaign = Campaign(
            searchspace=SearchSpace.from_product(parameters=parameters),
            objective=objective,
            **recommender_dct,
        )
        scenarios[scenario_name] = campaign

    return simulate_scenarios(
        scenarios,
        lookup,
        batch_size=scenario_config.batch_size,
        n_doe_iterations=scenario_config.n_doe_iterations,
        n_mc_iterations=scenario_config.n_mc_iterations,
        impute_mode="error",
    )


benchmark_config = RecommenderConvergenceAnalysis(
    recommenders={
        "Default Recommender": DEFAULT_RECOMMENDER,
        "Random Recommender": RandomRecommender(),
    },
    batch_size=5,
    n_doe_iterations=30,
    n_mc_iterations=50,
)

benchmark = Benchmark(
    name="synthetic_2C1D_1C",
    settings=benchmark_config,
    identifier=UUID("4e131cb7-4de0-4900-b993-1d7d4a194532"),
    callable=benchmark_callable,
)


if __name__ == "__main__":
    #  Visualize the domain

    import matplotlib.pyplot as plt

    X = np.linspace(-2 * pi, 2 * pi)
    Y = np.linspace(-2 * pi, 2 * pi)
    Z = [1, 2, 3, 4]

    x_mesh, y_mesh = np.meshgrid(X, Y)

    fig = plt.figure(figsize=(10, 10))
    for i, z in enumerate(Z):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        z_mesh = lookup(x_mesh, y_mesh, z)
        ax.plot_surface(x_mesh, y_mesh, z_mesh)
        plt.title(f"{z=}")

    plt.show()
