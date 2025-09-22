"""Easom function with transfer learning, negated function and a noise_std of 0.05."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from baybe.campaign import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalDiscreteParameter, TaskParameter
from baybe.parameters.base import DiscreteParameter
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget
from baybe.utils.random import temporary_seed
from benchmarks.definition import (
    ConvergenceBenchmark,
    ConvergenceBenchmarkSettings,
)
from benchmarks.definition.base import RunMode


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


def easom_tl_47_negate_noise5(settings: ConvergenceBenchmarkSettings) -> pd.DataFrame:
    """Benchmark function for transfer learning with the Easom function.

    Key characteristics:
    • Compares two negated versions of Easom function:
      - Target: standard negated Easom
      - Source: negated Easom with added noise (noise_std=0.05)
    • Uses 47 points per dimension
    • Tests transfer learning with different source data percentages:
      - 1% of source data
      - 5% of source data
      - 10% of source data
      - 20% of source data

    Args:
        settings: Configuration settings for the convergence benchmark

    Returns:
        DataFrame containing benchmark results
    """
    points_per_dim = 47
    negate = True
    percentages = [0.01, 0.05, 0.1]
    bounds: np.ndarray = np.array([[-10] * 2, [10] * 2])
    grid_locations = {
        f"x{d}": np.linspace(lower, upper, points_per_dim)
        for d, (lower, upper) in enumerate(bounds.T)
    }

    functions = {
        "Target_Function": lambda x: easom(x, negate=negate),
        "Source_Function": lambda x: easom(x, noise_std=0.05, negate=negate),
    }

    params: list[DiscreteParameter] = [
        NumericalDiscreteParameter(name=name, values=values)
        for name, values in grid_locations.items()
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
        target=NumericalTarget(name="Target", minimize=not negate)
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

    with temporary_seed(settings.random_seed):
        lookups = []
        for function_name, function in functions.items():
            lookup = pd.DataFrame(
                {f"x{d}": grid_d.ravel() for d, grid_d in enumerate(meshgrid)}
            )
            lookup["Target"] = lookup.apply(
                function, axis=1
            )  # Randomness from source function
            lookup["Function"] = function_name
            lookups.append(lookup)
        concat_lookups = pd.concat(lookups)

    lookup = concat_lookups[concat_lookups["Function"] == "Target_Function"]
    initial_data = concat_lookups[concat_lookups["Function"] == "Source_Function"]

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
        RunMode.DEFAULT: 30,
        RunMode.SMOKETEST: 2,
    },
    n_mc_iterations_settings={
        RunMode.DEFAULT: 100,
        RunMode.SMOKETEST: 2,
    },
)

easom_tl_47_negate_noise5_benchmark = ConvergenceBenchmark(
    function=easom_tl_47_negate_noise5,
    optimal_target_values={"Target": 0.9635009628660742},
    settings=benchmark_config,
)
