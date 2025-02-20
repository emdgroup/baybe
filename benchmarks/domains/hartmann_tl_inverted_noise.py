"""Transfer learning benchmark with inverted Hartmann functions as tasks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from botorch.test_functions.synthetic import Hartmann

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

DIMENSION = 3  # input dimensionality of the test function
POINTS_PER_DIM = 5  # number of grid points per input dimension
BOUNDS = Hartmann(dim=DIMENSION).bounds


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
    test_functions = {
        "Test_Function": lambda x: Hartmann(dim=DIMENSION, negate=True)
        .forward(torch.tensor(x))
        .item(),
        "Training_Function": lambda x: Hartmann(
            dim=DIMENSION, negate=False, noise_std=0.15
        )
        .forward(torch.tensor(x))
        .item(),
    }

    grid = np.meshgrid(*[points for points in _grid_locations().values()])

    lookups = []
    for function_name, function in test_functions.items():
        lookup = pd.DataFrame(
            {f"x{d}": grid_d.ravel() for d, grid_d in enumerate(grid)}
        )
        lookup["Target"] = lookup.apply(function, axis=1)
        lookup["Function"] = function_name
        lookups.append(lookup)
    lookups = pd.concat(lookups)

    return lookups


data = get_data()

test_task = "Test_Function"
source_task = "Training_Function"


def space_data() -> (
    SingleTargetObjective,
    SearchSpace,
    SearchSpace,
    pd.DataFrame,
    pd.DataFrame,
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
        values=[test_task, source_task],
        active_values=[test_task],
    )

    objective = SingleTargetObjective(target=NumericalTarget(name="Target", mode="MAX"))
    searchspace = SearchSpace.from_product(parameters=[*data_params, task_param])
    searchspace_nontl = SearchSpace.from_product(parameters=data_params)

    lookup = data.query(f'Function=="{test_task}"').copy(deep=True)
    initial_data = data.query(f'Function=="{source_task}"', engine="python").copy(
        deep=True
    )

    return objective, searchspace, searchspace_nontl, initial_data, lookup


def hartmann_tl_inverted_noise(settings: ConvergenceBenchmarkSettings) -> pd.DataFrame:
    """Benchmark function comparing TL and non-TL campaigns.

    Inputs:
        x0  Discrete numerical parameter [0,1]
        x1  Discrete numerical parameter [0,1]
        x2  Discrete numerical parameter [0,1]
        Function  Discrete task parameter
    Output: continuous
    Objective: Maximization
    Optimal Inputs: [
        {
            x0 0.25
            x1 0.6
            x2 0.75
        }
    ]
    Optimal Output: 2.999716768817375
    """
    objective, searchspace, searchspace_nontl, initial_data, lookup = space_data()

    campaign = Campaign(
        searchspace=searchspace,
        objective=objective,
    )

    results = []
    for p in [0.01, 0.02, 0.05, 0.1, 0.2]:
        results.append(
            simulate_scenarios(
                {f"{int(100 * p)}": campaign},
                lookup,
                initial_data=[
                    initial_data.sample(frac=p) for _ in range(settings.n_mc_iterations)
                ],
                batch_size=settings.batch_size,
                n_doe_iterations=settings.n_doe_iterations,
                impute_mode="error",
            )
        )
    # No training data
    results.append(
        simulate_scenarios(
            {"0": campaign},
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
    n_mc_iterations=100,
)

hartmann_tl_inverted_noise_benchmark = ConvergenceBenchmark(
    function=hartmann_tl_inverted_noise,
    optimal_target_values={"Target": 2.999716768817375},
    settings=benchmark_config,
)
