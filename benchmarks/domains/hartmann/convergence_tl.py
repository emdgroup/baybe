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
from baybe.utils.random import temporary_seed
from benchmarks.definition import (
    ConvergenceBenchmark,
    ConvergenceBenchmarkSettings,
)
from benchmarks.definition.base import RunMode
from benchmarks.domains.hartmann.utils import CustomHartmann


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
    return _compose_hartmann_tl_3_20_15(
        settings=settings,
        source_noise_std=0.15,
        source_shift=None,
        source_negate=False,
    )


def hartmann_tl_inv_3_20_15(settings: ConvergenceBenchmarkSettings) -> pd.DataFrame:
    """Benchmark function for transfer learning with inverted Hartmann function in 3D.

    Key characteristics:
    • Compares two versions of Hartmann function:
      - Target: standard Hartmann
      - Source: Inverted Hartmann with added noise (noise_std=0.15)
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
    return _compose_hartmann_tl_3_20_15(
        settings=settings,
        source_noise_std=0.15,
        source_shift=None,
        source_negate=True,
    )


def hartmann_tl_shift_3_20_15(settings: ConvergenceBenchmarkSettings) -> pd.DataFrame:
    """Benchmark function for transfer learning with shifted input Hartmann in 3D.

    Key characteristics:
    • Compares two versions of Hartmann function:
      - Target: standard Hartmann
      - Source: Shifted Hartmann (shifted first input dimension)
        with added noise (noise_std=0.15)
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
    return _compose_hartmann_tl_3_20_15(
        settings=settings,
        source_noise_std=0.15,
        source_shift=[0.2, 0, 0],
        source_negate=False,
    )


def _compose_hartmann_tl_3_20_15(
    settings: ConvergenceBenchmarkSettings,
    source_noise_std: float,
    source_shift: list[float] | None,
    source_negate: bool,
) -> pd.DataFrame:
    """Construct benchmark for transfer learning with the Hartmann function in 3D.

    Key characteristics:
    • Compares two versions of Hartmann functions (source and target)
    • Uses 20 points per dimension
    • Tests transfer learning with different source data percentages:
      - 1% of source data
      - 10% of source data
      - 20% of source data

    Args:
        settings: Configuration settings for the convergence benchmark.
        source_noise_std: Standard deviation of Gaussian noise to add to
            source function outputs.
        source_shift: Amount to shift individual dimension coordinates in
            source function. E.g. [0.2, 0, 0] would shift dimension 0 by 0.2.
            If None, no shifting is applied.
        source_negate: If True, negate the output of the source function.

    Returns:
        DataFrame containing benchmark results.

    Raises:
        ValueError: If source_shift is provided but does not have length 3.
    """
    if source_shift is not None and len(source_shift) != 3:
        raise ValueError("Shift list must have length 3 for 3D Hartmann function.")

    # Define base bounds
    bounds = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]).T

    # Create source function with specified parameters
    source_function = CustomHartmann(
        bounds=bounds,
        shift=source_shift,
        dim=3,
        noise_std=source_noise_std,
        negate=source_negate,
    )

    # Create target function (standard Hartmann with adjusted bounds from source)
    target_function = Hartmann(dim=source_function.dim, bounds=source_function._bounds)

    points_per_dim = 20
    percentages = [0.01, 0.05, 0.1]

    # Create grid locations for the parameters
    grid_locations = {
        f"x{d}": np.linspace(lower, upper, points_per_dim)
        for d, (lower, upper) in enumerate(bounds)
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

    with temporary_seed(settings.random_seed):
        target_values_tensor = source_function(
            initial_data_tensor
        )  # Randomness from source function

    # Assign the results back to a new DataFrame for initial_data
    initial_data["Target"] = target_values_tensor.detach().numpy()
    initial_data["Function"] = "Source_Function"

    lookup = arrays_to_dataframes([p.name for p in params], ["Target"], use_torch=True)(
        target_function
    )

    initial_data_samples = {}
    with temporary_seed(settings.random_seed):
        for p in percentages:
            initial_data_samples[p] = [
                initial_data.sample(frac=p) for _ in range(settings.n_mc_iterations)
            ]

    results = []
    for p in percentages:
        results.append(
            simulate_scenarios(
                {
                    f"{int(100 * p)}": tl_campaign,
                    f"{int(100 * p)}_naive": nontl_campaign,
                },
                lookup,
                initial_data=initial_data_samples[p],
                batch_size=settings.batch_size,
                n_doe_iterations=settings.n_doe_iterations,
                impute_mode="error",
                random_seed=settings.random_seed,
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
            random_seed=settings.random_seed,
        )
    )
    return pd.concat(results)


benchmark_config = ConvergenceBenchmarkSettings(
    batch_size_settings={
        RunMode.DEFAULT: 2,
        RunMode.SMOKETEST: 2,
    },
    n_doe_iterations_settings={
        RunMode.DEFAULT: 25,
        RunMode.SMOKETEST: 2,
    },
    n_mc_iterations_settings={
        RunMode.DEFAULT: 75,
        RunMode.SMOKETEST: 2,
    },
)

hartmann_tl_3_20_15_benchmark = ConvergenceBenchmark(
    function=hartmann_tl_3_20_15,
    optimal_target_values={"Target": -3.8324342572721695},
    settings=benchmark_config,
)

hartmann_tl_inv_3_20_15_benchmark = ConvergenceBenchmark(
    function=hartmann_tl_inv_3_20_15,
    optimal_target_values={"Target": -3.8324342572721695},
    settings=benchmark_config,
)

hartmann_tl_shift_3_20_15_benchmark = ConvergenceBenchmark(
    function=hartmann_tl_shift_3_20_15,
    optimal_target_values={"Target": -3.8324342572721695},
    settings=benchmark_config,
)
