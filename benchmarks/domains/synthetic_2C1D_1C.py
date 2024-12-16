"""Synthetic function with two continuous and one discrete input."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy import pi, sin, sqrt
from pandas import DataFrame

from baybe.campaign import Campaign
from baybe.parameters import NumericalContinuousParameter, NumericalDiscreteParameter
from baybe.recommenders import RandomRecommender
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget
from benchmarks.definition import (
    Benchmark,
    ConvergenceExperimentSettings,
)

if TYPE_CHECKING:
    from mpl_toolkits.mplot3d import Axes3D


def _lookup(arr: np.ndarray, /) -> np.ndarray:
    """Numpy-based lookup callable defining the objective function."""
    x, y, z = np.array_split(arr, 3, axis=1)
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


def lookup(df: pd.DataFrame, /) -> pd.DataFrame:
    """Dataframe-based lookup callable used as the loop-closing element."""
    return pd.DataFrame(
        _lookup(df[["x", "y", "z"]].to_numpy()), columns=["target"], index=df.index
    )


def synthetic_2C1D_1C(settings: ConvergenceExperimentSettings) -> DataFrame:
    """Hybrid synthetic test function.

    Inputs:
        z   discrete   {1,2,3,4}
        x   continuous [-2*pi, 2*pi]
        y   continuous [-2*pi, 2*pi]
    Output: continuous
    Objective: Maximization
    Optimal Inputs:
        {x: 1.610, y: 1.571, z: 3}
        {x: 1.610, y: -4.712, z: 3}
    Optimal Output: 4.09685
    """
    parameters = [
        NumericalContinuousParameter("x", (-2 * pi, 2 * pi)),
        NumericalContinuousParameter("y", (-2 * pi, 2 * pi)),
        NumericalDiscreteParameter("z", (1, 2, 3, 4)),
    ]

    target = NumericalTarget(name="target", mode="MAX")
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


benchmark_config = ConvergenceExperimentSettings(
    batch_size=5,
    n_doe_iterations=30,
    n_mc_iterations=50,
)

synthetic_2C1D_1C_benchmark = Benchmark(
    function=synthetic_2C1D_1C,
    best_possible_result=4.09685,
    settings=benchmark_config,
    optimal_function_inputs=[
        {"x": 1.610, "y": 1.571, "z": 3},
        {"x": 1.610, "y": -4.712, "z": 3},
    ],
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
        ax: Axes3D = fig.add_subplot(2, 2, i + 1, projection="3d")
        t_mesh = _lookup(
            np.c_[x_mesh.ravel(), y_mesh.ravel(), np.repeat(z, x_mesh.size)]
        ).reshape(x_mesh.shape)
        ax.plot_surface(x_mesh, y_mesh, t_mesh)
        plt.title(f"{z=}")

    plt.show()
