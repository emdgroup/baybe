"""5-dimensional Michalewicz function in a continuous space."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy import pi, sin
from pandas import DataFrame

from baybe.campaign import Campaign
from baybe.parameters import NumericalContinuousParameter
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
    try:
        assert np.all((arr >= 0) & (arr <= pi))
    except AssertionError:
        raise ValueError("Inputs are not in the valid ranges.")
    x1, x2, x3, x4, x5 = np.array_split(arr, 5, axis=1)

    return -(
        sin(x1) * sin(1 * x1**2 / pi) ** (2 * 10)
        + sin(x2) * sin(2 * x2**2 / pi) ** (2 * 10)
        + sin(x3) * sin(3 * x3**2 / pi) ** (2 * 10)
        + sin(x4) * sin(4 * x4**2 / pi) ** (2 * 10)
        + sin(x5) * sin(5 * x5**2 / pi) ** (2 * 10)
    )


def lookup(df: pd.DataFrame, /) -> pd.DataFrame:
    """Dataframe-based lookup callable used as the loop-closing element."""
    return pd.DataFrame(
        _lookup(df[["x1", "x2", "x3", "x4", "x5"]].to_numpy()),
        columns=["target"],
        index=df.index,
    )


def synthetic_michalewicz(settings: ConvergenceExperimentSettings) -> DataFrame:
    """5-dimensional Michalewicz function.

    Details of the function can be found at https://www.sfu.ca/~ssurjano/michal.html

    Inputs:
        x1,...,x5   continuous   [0, pi]
    Output: continuous
    Objective: Minimization
    Optimal Input:
        {x1: 2.203,  x2: 1.571, x3: 1.285,  x4: 1.923, x5: 1.720e}
    Optimal Output: -4.687658
    """
    parameters = [
        NumericalContinuousParameter(name=f"x{i}", bounds=(0, pi)) for i in range(1, 6)
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


benchmark_config = ConvergenceExperimentSettings(
    batch_size=5,
    n_doe_iterations=25,
    n_mc_iterations=20,
)

synthetic_michalewicz_benchmark = Benchmark(
    function=synthetic_michalewicz,
    best_possible_result=-4.687658,
    settings=benchmark_config,
)

if __name__ == "__main__":
    #  Visualization of the 2-dimensional variant

    import matplotlib.pyplot as plt

    X1 = np.linspace(0, pi, 50)
    X2 = np.linspace(0, pi, 50)
    X1, X2 = np.meshgrid(X1, X2)

    # Michalewicz function
    Z = -1 * (
        (np.sin(X1) * np.sin((1 * X1**2) / np.pi) ** 20)
        + (np.sin(X2) * np.sin((2 * X2**2) / np.pi) ** 20)
    )

    ax: Axes3D = plt.figure().add_subplot(projection="3d")
    surf = ax.plot_surface(X1, X2, Z)

    ax.set_xlabel("x1", fontsize=10)
    ax.set_ylabel("x2", fontsize=10)
    ax.tick_params(axis="both", which="major", labelsize=6)

    plt.show()
