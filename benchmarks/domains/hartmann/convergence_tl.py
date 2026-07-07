"""Transfer learning benchmark with inverted Hartmann functions as tasks."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from botorch.test_functions.synthetic import Hartmann

from baybe.campaign import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalDiscreteParameter, TaskParameter
from baybe.parameters.base import DiscreteParameter
from baybe.searchspace import SearchSpace
from baybe.settings import Settings
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget
from baybe.utils.dataframe import arrays_to_dataframes
from benchmarks.definition import (
    ConvergenceBenchmark,
    ConvergenceBenchmarkSettings,
)
from benchmarks.definition.base import RunMode
from benchmarks.domains.hartmann.utils import ShiftedHartmann


def _make_hartmann_tl_benchmark(
    name: str,
    *,
    source_noise_std: float,
    source_shift: tuple[float, float, float] | None,
    source_negate: bool,
) -> Callable[[ConvergenceBenchmarkSettings], pd.DataFrame]:
    """Return a named Hartmann transfer-learning benchmark callable.

    The benchmark operates on Hartmann function in 3D.
    It compares two discretized versions of the Hartmann function:
      - Target: standard Hartmann
      - Source: Hartmann with optional changes (noise, shifting, or negation)
    - Uses 20 points per dimension
    - Tests transfer learning with different source data percentages:
      - 1% of source data
      - 10% of source data
      - 20% of source data

    The callable requires one argument:
        settings: Configuration settings for the convergence benchmark.
    The callable returns:
        DataFrame containing benchmark results.

    Args:
        name: Benchmark name.
        source_noise_std: Noise added to the source Hartmann function.
        source_shift: Shift added to the source Hartmann function.
        source_negate: Whether to negate the source Hartmann function.

    Returns:
        The callable returning the benchmark results.

    Raises:
        ValueError: If ``source_shift`` is provided but does not have length 3.
    """
    if source_shift is not None and len(source_shift) != 3:
        raise ValueError("Shift list must have length 3 for 3D Hartmann function.")

    def benchmark_fn(settings: ConvergenceBenchmarkSettings) -> pd.DataFrame:
        """Execute a Hartmann transfer-learning benchmark variant."""
        # Define base bounds
        bounds = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]).T

        # Create source function with specified parameters
        source_function = ShiftedHartmann(
            bounds=bounds,
            shift=list(source_shift) if source_shift is not None else None,
            dim=3,
            noise_std=source_noise_std,
            negate=source_negate,
        )

        # Create target function (standard Hartmann with adjusted bounds from source)
        target_function = Hartmann(
            dim=source_function.dim, bounds=source_function._bounds
        )

        points_per_dim = 20
        percentages = [0.01, 0.05, 0.1]

        # Create grid locations for the parameters
        grid_locations = {
            f"x{d}": np.linspace(lower, upper, points_per_dim)
            for d, (lower, upper) in enumerate(bounds)
        }

        params: list[DiscreteParameter] = [
            NumericalDiscreteParameter(
                name=n,
                values=tuple(points),
            )
            for n, points in grid_locations.items()
        ]
        searchspace_nontl = SearchSpace.from_product(parameters=params)
        tl_searchspace = SearchSpace.from_product(
            parameters=params
            + [
                TaskParameter(
                    name="Function",
                    values=("Target_Function", "Source_Function"),
                    active_values=("Target_Function",),
                )
            ]
        )

        objective = SingleTargetObjective(
            target=NumericalTarget(name="Target", minimize=True)
        )
        tl_campaign = Campaign(searchspace=tl_searchspace, objective=objective)
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

        with Settings(random_seed=settings.random_seed):
            target_values_tensor = source_function(
                initial_data_tensor
            )  # Randomness from source function

        # Assign the results back to a new DataFrame for initial_data
        initial_data["Target"] = target_values_tensor.detach().numpy()
        initial_data["Function"] = "Source_Function"

        lookup = arrays_to_dataframes(
            [p.name for p in params], ["Target"], use_torch=True
        )(target_function)

        initial_data_samples = {}
        with Settings(random_seed=settings.random_seed):
            for p in percentages:
                initial_data_samples[p] = [
                    initial_data.sample(frac=p) for _ in range(settings.n_mc_iterations)
                ]

        results = []
        for p in percentages:
            scenarios = {
                f"{int(100 * p)}_tl": tl_campaign,
                f"{int(100 * p)}_naive": nontl_campaign,
            }
            results.append(
                simulate_scenarios(
                    scenarios,
                    lookup,
                    initial_data=initial_data_samples[p],
                    batch_size=settings.batch_size,
                    n_doe_iterations=settings.n_doe_iterations,
                    impute_mode="error",
                    random_seed=settings.random_seed,
                )
            )
        scenarios_0 = {
            "0_tl": tl_campaign,
            "0_naive": nontl_campaign,
        }
        results.append(
            simulate_scenarios(
                scenarios_0,
                lookup,
                batch_size=settings.batch_size,
                n_doe_iterations=settings.n_doe_iterations,
                n_mc_iterations=settings.n_mc_iterations,
                impute_mode="error",
                random_seed=settings.random_seed,
            )
        )
        return pd.concat(results)

    benchmark_fn.__name__ = name
    benchmark_fn.__qualname__ = name
    return benchmark_fn


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
    function=_make_hartmann_tl_benchmark(
        name="hartmann_tl_3_20_15",
        source_noise_std=0.15,
        source_shift=None,
        source_negate=False,
    ),
    optimal_target_values={"Target": -3.8324342572721695},
    settings=benchmark_config,
)

hartmann_tl_inv_3_20_15_benchmark = ConvergenceBenchmark(
    function=_make_hartmann_tl_benchmark(
        name="hartmann_tl_inv_3_20_15",
        source_noise_std=0.15,
        source_shift=None,
        source_negate=True,
    ),
    optimal_target_values={"Target": -3.8324342572721695},
    settings=benchmark_config,
)

hartmann_tl_shift_3_20_15_benchmark = ConvergenceBenchmark(
    function=_make_hartmann_tl_benchmark(
        "hartmann_tl_shift_3_20_15",
        source_noise_std=0.15,
        source_shift=(0.2, 0, 0),
        source_negate=False,
    ),
    optimal_target_values={"Target": -3.8324342572721695},
    settings=benchmark_config,
)
