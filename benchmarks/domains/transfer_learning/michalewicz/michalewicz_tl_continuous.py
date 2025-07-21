"""Transfer learning in a continuous space with negated Michalewicz functions as tasks.

This uses the 5-dimensional version of the Michalewictz function and adds
a noise_std of 0.15 to the source function.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from botorch.test_functions.synthetic import Michalewicz

from baybe.campaign import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalContinuousParameter, TaskParameter
from baybe.parameters.base import Parameter
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget
from benchmarks.definition import ConvergenceBenchmark, ConvergenceBenchmarkSettings


def make_searchspace(use_task_parameter: bool) -> SearchSpace:
    """Create search space for the benchmark."""
    params: list[Parameter] = [
        NumericalContinuousParameter(
            name=f"x{k}",
            bounds=(0, math.pi),
        )
        for k in range(5)
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


def make_objective() -> SingleTargetObjective:
    """Create the objective for the benchmark."""
    return SingleTargetObjective(target=NumericalTarget(name="Target"))


def wrap_function(
    function: Callable, function_name: str, df: pd.DataFrame
) -> pd.DataFrame:
    """Wrap the given function such that it operates on DataFrames."""
    input_columns = df.columns.tolist()

    # Create a copy to avoid modifying the original DataFrame
    result_df = df.copy()

    # Apply Michalewicz function to each row
    def apply_function(row):
        # Remove the "Function" column but record its value such that it can be added
        # back to the row after the function evaluation
        if "Function" in row.index:
            function_value = row["Function"]
            row = row.drop("Function")
        else:
            function_value = None  # Default if column doesn't exist

        x = torch.tensor(row.values.astype(float), dtype=torch.float64).unsqueeze(0)
        # Re-add the "Function" column to the row
        if function_value is not None:
            row["Function"] = function_value
        return function(x).item()

    result_df["Target"] = (
        result_df[input_columns].apply(apply_function, axis=1).astype(float)
    )
    # Add a column "Function" with the function name
    result_df["Function"] = function_name

    return result_df


def make_initial_data(
    function: Callable, function_name: str, num_of_points: int
) -> pd.DataFrame:
    """Create initial data points for the Michalewicz benchmark."""
    # Create random samples in [0, pi]^dim
    samples = np.random.uniform(low=0, high=math.pi, size=(num_of_points, 5))

    # Convert to DataFrame
    column_names = [f"x{i}" for i in range(5)]
    df = pd.DataFrame(samples, columns=column_names)

    # Apply the function to get target values
    df = wrap_function(function, function_name, df)

    return df


def michalewicz_tl_continuous(settings: ConvergenceBenchmarkSettings) -> pd.DataFrame:
    """Benchmark function for transfer learning with the Michalewicz function in 5D.

    Key characteristics:
    • Compares transfer learning vs. non-transfer learning approaches
    • Uses two versions of Michalewicz function:
      - Target: standard negated Michalewicz
      - Source: negated Michalewicz with added noise (noise_std=0.15)
    • Tests varying amounts of source data (1, 10, 50, 100 points)
    • Includes baseline with no transfer learning (0 points)
    • Creates two campaign types:
      - Transfer learning: with task parameter to distinguish data sources
      - Non-transfer learning (naive): without task parameter

    Args:
        settings: Configuration settings for the convergence benchmark

    Returns:
        DataFrame containing benchmark results for all test cases
    """
    functions = {
        "Source_Function": Michalewicz(dim=5, negate=True, noise_std=0.15),
        "Target_Function": Michalewicz(dim=5, negate=True),
    }
    searchspace_nontl = make_searchspace(use_task_parameter=False)
    searchspace_tl = make_searchspace(use_task_parameter=True)

    objective = make_objective()
    campaign_tl = Campaign(
        searchspace=searchspace_tl,
        objective=objective,
    )
    campaign_nontl = Campaign(
        searchspace=searchspace_nontl,
        objective=objective,
    )

    results = []

    for p in [1, 10, 50, 100]:
        results.append(
            simulate_scenarios(
                {f"{p}": campaign_tl, f"{p}_naive": campaign_nontl},
                lambda x: wrap_function(
                    functions["Target_Function"], "Target_Function", x
                ),
                initial_data=[
                    make_initial_data(
                        functions["Source_Function"], "Source_Function", p
                    )
                    for _ in range(settings.n_mc_iterations)
                ],
                batch_size=settings.batch_size,
                n_doe_iterations=settings.n_doe_iterations,
                impute_mode="error",
            )
        )
    results.append(
        simulate_scenarios(
            {"0": campaign_tl, "0_naive": campaign_nontl},
            lambda x: wrap_function(functions["Target_Function"], "Target_Function", x),
            batch_size=settings.batch_size,
            n_doe_iterations=settings.n_doe_iterations,
            n_mc_iterations=settings.n_mc_iterations,
            impute_mode="error",
        )
    )
    return pd.concat(results)


benchmark_config = ConvergenceBenchmarkSettings(
    batch_size=2,
    n_doe_iterations=30,
    n_mc_iterations=30,
)

michalewicz_tl_continuous_benchmark = ConvergenceBenchmark(
    function=michalewicz_tl_continuous,
    optimal_target_values={"Target": 4.687658},
    settings=benchmark_config,
)
