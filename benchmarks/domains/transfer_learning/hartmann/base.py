"""Transfer learning benchmark with inverted Hartmann functions as tasks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from botorch.test_functions.synthetic import Hartmann

from baybe.campaign import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalDiscreteParameter, TaskParameter
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget
from benchmarks.definition import (
    ConvergenceBenchmarkSettings,
)


def grid_locations(points_per_dim: int, dim: int) -> dict[str, np.array]:
    """Locations of measurements for every dimension.

    Args:
        points_per_dim: Number of grid points per input dimension.
        dim: The input dimension of the Hartmann function.

    Returns:
        Dictionary with dimension names (keys) and corresponding measurement points.
    """
    bounds = Hartmann(dim=dim).bounds
    return {
        f"x{d}": np.linspace(lower, upper, points_per_dim)
        for d, (lower, upper) in enumerate(bounds.T)
    }


def get_data(functions: dict[str, callable], grid: dict[str, np.array]) -> pd.DataFrame:
    """Generate data for benchmark."""
    grid = np.meshgrid(*[points for points in grid.values()])

    lookups = []
    for function_name, function in functions.items():
        lookup = pd.DataFrame(
            {f"x{d}": grid_d.ravel() for d, grid_d in enumerate(grid)}
        )
        lookup["Target"] = lookup.apply(function, axis=1)
        lookup["Function"] = function_name
        lookups.append(lookup)
    lookups = pd.concat(lookups)

    return lookups


def create_searchspace(
    grid_locations: dict[str, np.array], use_task_parameter: bool
) -> SearchSpace:
    """Create search space for the benchmark."""
    params = [
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


def create_objective(negate: bool) -> SingleTargetObjective:
    """Create the objective for the benchmark."""
    return SingleTargetObjective(
        target=NumericalTarget(name="Target", mode="MAX" if negate else "MIN")
    )


def create_lookup(data: pd.DataFrame) -> pd.DataFrame:
    """Create the lookup for the benchmark."""
    return data[data["Function"] == "Target_Function"]


def create_initial_data(data: pd.DataFrame) -> pd.DataFrame:
    """Create the initial data for the benchmark."""
    return data[data["Function"] == "Source_Function"]


def abstract_hartmann_tl_noise(
    settings: ConvergenceBenchmarkSettings,
    functions: dict[str, callable],
    points_per_dim: int,
    dim: int,
    percentages: list[float],
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
    grid = grid_locations(points_per_dim, dim)
    data = get_data(functions, grid)

    searchspace_nontl = create_searchspace(grid, use_task_parameter=False)
    searchspace_tl = create_searchspace(grid, use_task_parameter=True)

    initial_data = create_initial_data(data)
    lookup = create_lookup(data)

    objective = create_objective(negate)
    campaign_tl = Campaign(
        searchspace=searchspace_tl,
        objective=objective,
    )
    campaign_nontl = Campaign(
        searchspace=searchspace_nontl,
        objective=objective,
    )

    results = []
    for p in percentages:
        results.append(
            simulate_scenarios(
                {f"{int(100 * p)}": campaign_tl},
                lookup,
                initial_data=[
                    initial_data.sample(frac=p) for _ in range(settings.n_mc_iterations)
                ],
                batch_size=settings.batch_size,
                n_doe_iterations=settings.n_doe_iterations,
                impute_mode="error",
            )
        )
    # No training data and non-TL campaign
    results.append(
        simulate_scenarios(
            {"0": campaign_tl, "non-TL": campaign_nontl},
            lookup,
            batch_size=settings.batch_size,
            n_doe_iterations=settings.n_doe_iterations,
            n_mc_iterations=settings.n_mc_iterations,
            impute_mode="error",
        )
    )
    results = pd.concat(results)
    return results
