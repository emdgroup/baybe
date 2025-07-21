"""Hartmann function in 3 dimensions as a benchmark."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
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

if TYPE_CHECKING:
    from mpl_toolkits.mplot3d import Axes3D


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
    )


benchmark_config = ConvergenceBenchmarkSettings(
    batch_size=5,
    n_doe_iterations=30,
    n_mc_iterations=100,
)

hartmann_3d_benchmark = ConvergenceBenchmark(
    function=hartmann_3d,
    optimal_target_values={"target": -3.86278},
    settings=benchmark_config,
)


if __name__ == "__main__":
    # Visualize the domain (limited to 2D slices due to 3D nature)

    import matplotlib.pyplot as plt

    # Create a grid of points for x1 and x2 dimensions
    x1 = np.linspace(0, 1, 50)
    x2 = np.linspace(0, 1, 50)
    x1_mesh, x2_mesh = np.meshgrid(x1, x2)

    # Create slices for different fixed values of x3
    x3_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    # Instantiate the Hartmann function
    hartmann_func = Hartmann(dim=3, negate=False)

    fig = plt.figure(figsize=(15, 10))
    for i, x3_val in enumerate(x3_values):
        ax: Axes3D = fig.add_subplot(2, 3, i + 1, projection="3d")
        x3_mesh = np.full_like(x1_mesh, x3_val)

        # Stack the meshgrid points into the required shape
        points = np.column_stack((x1_mesh.ravel(), x2_mesh.ravel(), x3_mesh.ravel()))

        # Calculate function values
        import torch

        with torch.no_grad():
            values = (
                hartmann_func(torch.tensor(points, dtype=torch.float64))
                .numpy()
                .reshape(x1_mesh.shape)
            )

        # Plot the surface
        surf = ax.plot_surface(x1_mesh, x2_mesh, values, cmap="viridis")

        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("Hartmann-3D")
        ax.set_title(f"x3 = {x3_val}")

    plt.tight_layout()
    plt.show()
