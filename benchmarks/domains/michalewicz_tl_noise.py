"""Transfer learning benchmark with noisy Michalewicz functions as tasks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from botorch.test_functions.synthetic import Michalewicz

from baybe.campaign import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalDiscreteParameter, TaskParameter
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget
from benchmarks.definition import (
    ConvergenceBenchmark,
    ConvergenceBenchmarkSettings,
)

DIMENSION = 4  # input dimensionality of the function
POINTS_PER_DIM = 15  # number of grid points per input dimension
BOUNDS = Michalewicz(dim=DIMENSION).bounds


def _grid_locations() -> dict[str, np.array]:
    """Locations of measurements for every dimension.

    Returns:
        Dictionary with dimension names (keys) and corresponding measurement
        points.
    """
    return {
        f"x{d}": np.linspace(lower, upper, POINTS_PER_DIM)
        for d, (lower, upper) in enumerate(BOUNDS.T)
    }


def get_data() -> pd.DataFrame:
    """Generate data for benchmark.

    Returns:
        Data for benchmark.
    """
    functions = {
        "Target_Function": lambda x: Michalewicz(dim=DIMENSION, negate=True)
        .forward(torch.tensor(x))
        .item(),
        "Source_Function": lambda x: Michalewicz(
            dim=DIMENSION, noise_std=0.15, negate=True
        )
        .forward(torch.tensor(x))
        .item(),
    }

    grid = np.meshgrid(*[points for points in _grid_locations().values()])

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


data = get_data()

target_task = "Target_Function"
source_task = "Source_Function"


def space_data() -> (
    tuple[
        SingleTargetObjective,
        SearchSpace,
        SearchSpace,
        pd.DataFrame,
        pd.DataFrame,
    ]
):
    """Definition of search space, objective, and data.

    Returns:
        Objective, TL search space, non-TL search space,
        pre-measured task data, and lookup for the active task
    """
    data_params = [
        NumericalDiscreteParameter(
            name=name,
            values=points,
        )
        for name, points in _grid_locations().items()
    ]

    task_param = TaskParameter(
        name="Function",
        values=[target_task, source_task],
        active_values=[target_task],
    )

    objective = SingleTargetObjective(target=NumericalTarget(name="Target", mode="MAX"))
    searchspace = SearchSpace.from_product(parameters=[*data_params, task_param])
    searchspace_nontl = SearchSpace.from_product(parameters=data_params)

    lookup = data.query(f'Function=="{target_task}"').copy(deep=True)
    initial_data = data.query(f'Function=="{source_task}"', engine="python").copy(
        deep=True
    )

    return objective, searchspace, searchspace_nontl, initial_data, lookup


def michalewicz_tl_noise(settings: ConvergenceBenchmarkSettings) -> pd.DataFrame:
    """Benchmark function comparing TL and non-TL campaigns.

    Inputs:
        x0  Discrete numerical parameter [0,3.1416]
        x1  Discrete numerical parameter [0,3.1416]
        x2  Discrete numerical parameter [0,3.1416]
        x3  Discrete numerical parameter [0,3.1416]
        Function  Discrete task parameter
    Output: continuous
    Objective: Maximization
    Optimal Inputs: [
        {
            x0: 2.243995,
            x1: 1.570796,
            x2: 1.346397,
            x3: 1.121997
        }
    ]
    Optimal Output: 3.418800985955677
    """
    objective, searchspace, searchspace_nontl, initial_data, lookup = space_data()

    campaign = Campaign(
        searchspace=searchspace,
        objective=objective,
    )

    results = []

    def sample_initial_data():
        p = 0.0001
        upsample_max_thr = 3
        n_upsample_max = 1
        data_sub = pd.concat(
            [
                # Sample specific fraction of initial data
                initial_data.sample(frac=p),
                # Add some points near optimum
                initial_data.query(
                    f"{objective._target.name}>{upsample_max_thr}"
                ).sample(n=n_upsample_max),
            ]
        )
        return data_sub

    results.append(
        simulate_scenarios(
            {"TL": campaign},
            lookup,
            initial_data=[
                sample_initial_data() for _ in range(settings.n_mc_iterations)
            ],
            batch_size=settings.batch_size,
            n_doe_iterations=settings.n_doe_iterations,
            impute_mode="error",
        )
    )
    # No training data
    results.append(
        simulate_scenarios(
            {"TL-noSource": campaign},
            lookup,
            batch_size=settings.batch_size,
            n_doe_iterations=settings.n_doe_iterations,
            n_mc_iterations=settings.n_mc_iterations,
            impute_mode="error",
        )
    )
    # Non-TL campaign
    results.append(
        simulate_scenarios(
            {"non-TL": Campaign(searchspace=searchspace_nontl, objective=objective)},
            lookup,
            batch_size=settings.batch_size,
            n_doe_iterations=settings.n_doe_iterations,
            n_mc_iterations=settings.n_mc_iterations,
            impute_mode="error",
        )
    )
    results = pd.concat(results)
    return results


benchmark_config = ConvergenceBenchmarkSettings(
    batch_size=2,
    n_doe_iterations=10,
    n_mc_iterations=50,
)

michalewicz_tl_noise_benchmark = ConvergenceBenchmark(
    function=michalewicz_tl_noise,
    optimal_target_values={"Target": 3.418800985955677},
    settings=benchmark_config,
)
