"""Transfer learning benchmark with inverted Hartmann functions as tasks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from botorch.test_functions.synthetic import Hartmann

from baybe.campaign import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalDiscreteParameter, TaskParameter
from baybe.parameters.base import DiscreteParameter
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget
from baybe.utils.dataframe import arrays_to_dataframes
from benchmarks.definition import (
    ConvergenceBenchmark,
    ConvergenceBenchmarkSettings,
)


def hartmann_tl_3_20_15(settings: ConvergenceBenchmarkSettings) -> pd.DataFrame:
    """Benchmark function for transfer learning with the Hartmann function in 3D.

    Key characteristics:
    • Compares two versions of Hartmann function:
      - Target: standard Hartmann
      - Source: Hartmann with added noise (noise_std=0.15)
    • Uses 20 points per dimension
    • Tests transfer learning with different source data percentages:
      - 1% of source data
      - 10% of source data
      - 20% of source data

    Args:
        settings: Configuration settings for the convergence benchmark.

    Returns:
        DataFrame containing benchmark results.
    """
    target_function = Hartmann(dim=3)
    source_function = Hartmann(dim=3, noise_std=0.15)

    points_per_dim = 20
    percentages = [0.01, 0.05, 0.1]

    # Create grid locations for the parameters
    bounds = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    grid_locations = {
        f"x{d}": np.linspace(lower, upper, points_per_dim)
        for d, (lower, upper) in enumerate(bounds.T)
    }

    params: list[DiscreteParameter] = [
        NumericalDiscreteParameter(
            name=name,
            values=points,
        )
        for name, points in grid_locations.items()
    ]
    task_param = TaskParameter(
        name="Function",
        values=["Target_Function", "Source_Function"],
        active_values=["Target_Function"],
    )
    params_tl = params + [task_param]

    searchspace_nontl = SearchSpace.from_product(parameters=params)
    searchspace_tl = SearchSpace.from_product(parameters=params_tl)

    objective = SingleTargetObjective(
        target=NumericalTarget(name="Target", minimize=True)
    )
    tl_campaign = Campaign(
        searchspace=searchspace_tl,
        objective=objective,
    )
    nontl_campaign = Campaign(
        searchspace=searchspace_nontl,
        objective=objective,
    )

    meshgrid = np.meshgrid(*[points for points in grid_locations.values()])

    # Create a DataFrame for the initial data coordinates
    coord_columns = [p.name for p in params]
    initial_data = pd.DataFrame(
        {f"x{d}": grid_d.ravel() for d, grid_d in enumerate(meshgrid)},
        columns=coord_columns,  # Ensure correct column order
    )

    # Convert coordinates to a PyTorch tensor
    initial_data_tensor = torch.tensor(initial_data[coord_columns].values)

    target_values_tensor = source_function(initial_data_tensor)

    # Assign the results back to a new DataFrame for initial_data
    initial_data["Target"] = target_values_tensor.detach().numpy()
    initial_data["Function"] = "Source_Function"

    lookup = arrays_to_dataframes([p.name for p in params], ["Target"], use_torch=True)(
        target_function
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


benchmark_config = ConvergenceBenchmarkSettings(
    batch_size=2,
    n_doe_iterations=25,
    n_mc_iterations=75,
)

hartmann_tl_3_20_15_benchmark = ConvergenceBenchmark(
    function=hartmann_tl_3_20_15,
    optimal_target_values={"Target": -3.851831124860353},
    settings=benchmark_config,
)
