"""Transfer learning benchmark with noisy Easom functions as tasks."""

from __future__ import annotations

import math
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
from benchmarks.definition import ConvergenceBenchmarkSettings


def grid_locations(points_per_dim: int) -> dict[str, np.ndarray]:
    """Locations of measurements for every dimension.

    Args:
        points_per_dim: Number of grid points per input dimension.

    Returns:
        Dictionary with dimension names (keys) and corresponding measurement points.
    """
    bounds: np.ndarray = np.array([[-10] * 2, [10] * 2])
    return {
        f"x{d}": np.linspace(lower, upper, points_per_dim)
        for d, (lower, upper) in enumerate(bounds.T)
    }


def easom(x: np.ndarray, noise_std: float = 0.0, negate: bool = False):
    """Eason function output values.

    Args:
        x: Input with two features
        noise_std: Noise to be added to output.
        negate: Whether to invert the output

    Returns:
        Easom function output
    """
    x = np.array(x).ravel()
    assert x.shape == (2,)
    y = (
        -math.cos(x[0])
        * math.cos(x[1])
        * math.exp(-((x[0] - math.pi) ** 2) - (x[1] - math.pi) ** 2)
    )
    if negate:
        y = y * -1
    if noise_std > 0:
        y += np.random.normal(loc=0.0, scale=noise_std, size=1)[0]
    return y


def generate_data(
    functions: dict[str, Callable], grid: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Generate data for benchmark.

    Args:
        functions: Dictionary of functions to generate data for. The keys
            need to be `Target_function` and `Source_function`.
        grid: Dictionary of grid locations for each dimension, created by
            `grid_locations()`.

    Returns:
        The data used in the benchmark.
    """
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


def easom_tl_noise(
    settings: ConvergenceBenchmarkSettings,
    functions: dict[str, Callable],
    points_per_dim: int,
    percentages: Iterable[float],
    negate: bool,
) -> pd.DataFrame:
    """Benchmark function comparing TL and non-TL campaigns.

    Inputs:
        x0  Discrete numerical parameter [-100,100]
        x1  Discrete numerical parameter [-100,100]
        Function  Discrete task parameter
    Output: continuous
    Objective: Maximization if `negate` else Minimization
    """
    grid = grid_locations(points_per_dim)
    data = generate_data(functions, grid)

    objective = SingleTargetObjective(
        target=NumericalTarget(name="Target", mode="MAX" if negate else "MIN")
    )
    searchspace_nontl = make_searchspace(grid, use_task_parameter=False)
    searchspace_tl = make_searchspace(grid, use_task_parameter=True)

    lookup = data[data["Function"] == "Target_Function"]
    initial_data = data[data["Function"] == "Source_Function"]

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
