"""Transfer learning benchmark with inverted Hartmann functions as tasks."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

from baybe.campaign import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalDiscreteParameter, TaskParameter
from baybe.parameters.base import DiscreteParameter
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget
from benchmarks.definition import (
    ConvergenceBenchmarkSettings,
)


def hartmann(X: np.ndarray, negate: bool, noise_std: float = 0) -> float:
    """Reimplementation of the three-dimensional Hartmann function."""
    x = np.array(X).ravel()
    A = np.array(
        [[3.0, 10.0, 30.0], [0.1, 10.0, 35.0], [3.0, 10.0, 30.0], [0.1, 10.0, 35.0]]
    )
    P = 1e-4 * np.array(
        [[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]]
    )
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    r = np.sum(A * np.square(x - P), axis=-1)
    factor = 1 if negate else -1
    h = factor * np.dot(np.exp(-r), alpha)
    if noise_std > 0:
        h += np.random.normal(loc=0.0, scale=noise_std, size=1)[0]
    return h


def grid_locations(points_per_dim: int) -> dict[str, np.ndarray]:
    """Locations of measurements for every dimension.

    Args:
        points_per_dim: Number of grid points per input dimension.

    Returns:
        Dictionary with dimension names (keys) and corresponding measurement points.
    """
    bounds = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    return {
        f"x{d}": np.linspace(lower, upper, points_per_dim)
        for d, (lower, upper) in enumerate(bounds.T)
    }


def generate_data(
    functions: dict[str, Callable], grid: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Generate data for benchmark."""
    meshgrid = np.meshgrid(*[points for points in grid.values()])

    lookups = []
    for function_name, function in functions.items():
        lookup = pd.DataFrame(
            {f"x{d}": grid_d.ravel() for d, grid_d in enumerate(meshgrid)}
        )
        lookup["Target"] = lookup.apply(function, axis=1)
        lookup["Function"] = function_name
        lookups.append(lookup)

    return pd.concat(lookups)


def make_searchspace(
    grid_locations: dict[str, np.ndarray], use_task_parameter: bool
) -> SearchSpace:
    """Create search space for the benchmark."""
    params: list[DiscreteParameter] = [
        NumericalDiscreteParameter(
            name=name,
            values=points,
        )
        for name, points in grid_locations.items()
    ]
    if use_task_parameter:
        params.append(
            TaskParameter(
                name="Function",
                values=["Target_Function", "Source_Function"],
                active_values=["Target_Function"],
            )
        )
    return SearchSpace.from_product(parameters=params)


def hartmann_tl_noise(
    settings: ConvergenceBenchmarkSettings,
    functions: dict[str, Callable],
    points_per_dim: int,
    percentages: Iterable[float],
    negate: bool,
) -> pd.DataFrame:
    """Benchmark function comparing TL and non-TL campaigns.

    Inputs:
        x0          Discrete numerical parameter [0,1]
        x1          Discrete numerical parameter [0,1]
        x2          Discrete numerical parameter [0,1]
        Function    Discrete task parameter
    Output: continuous

    """
    grid = grid_locations(points_per_dim)
    data = generate_data(functions, grid)

    searchspace_nontl = make_searchspace(grid, use_task_parameter=False)
    searchspace_tl = make_searchspace(grid, use_task_parameter=True)

    initial_data = data[data["Function"] == "Source_Function"]
    lookup = data[data["Function"] == "Target_Function"]

    objective = SingleTargetObjective(
        target=NumericalTarget(name="Target", mode="MAX" if negate else "MIN")
    )
    tl_campaign = Campaign(
        searchspace=searchspace_tl,
        objective=objective,
    )
    nontl_campaign = Campaign(
        searchspace=searchspace_nontl,
        objective=objective,
    )

    results = []
    for p in percentages:
        results.append(
            simulate_scenarios(
                {
                    f"{int(100 * p)}": tl_campaign,
                    f"{int(100 * p)}_naive": nontl_campaign,
                },
                lookup,
                initial_data=[
                    initial_data.sample(frac=p) for _ in range(settings.n_mc_iterations)
                ],
                batch_size=settings.batch_size,
                n_doe_iterations=settings.n_doe_iterations,
                impute_mode="error",
            )
        )
    results.append(
        simulate_scenarios(
            {"0": tl_campaign, "0_naive": nontl_campaign},
            lookup,
            batch_size=settings.batch_size,
            n_doe_iterations=settings.n_doe_iterations,
            n_mc_iterations=settings.n_mc_iterations,
            impute_mode="error",
        )
    )
    return pd.concat(results)
