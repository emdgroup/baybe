"""Benchmark on quadratic functions with transfer learning.

This module provides the basic structure for creating different
benchmarks by changing the source and target task configurations. The benchmark
compares TL and non-TL campaigns on synthetic quadratic functions: y = a*(x+b)^2 + c.

By convention, the benchmarks use descriptive names based on their configuration.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from baybe.campaign import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalDiscreteParameter, TaskParameter
from baybe.parameters.base import Parameter
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget
from benchmarks.definition import ConvergenceBenchmarkSettings


def get_optimal_target_value() -> float:
    """Calculate the theoretical optimal target value for the benchmark.

    Returns:
        Minimum value of the target quadratic function
    """
    return 0.0


def load_data(
    n_sources: int = 3, keep_min: bool = False, noise_std: float = 0.05
) -> pd.DataFrame:
    """Load synthetic quadratic data for transfer learning benchmarks.

    Creates source and target tasks using quadratic functions: y = a*(x+b)^2 + c
    For convergence benchmarks, we need a comprehensive lookup table.

    Args:
        n_sources: Number of source tasks to generate
        keep_min: If True, freeze b=0 for all functions (same minimum location)
        noise_std: Standard deviation of noise added to outputs (default: 0.05).

    Returns:
        DataFrame containing both source and target task data with columns:
        - x: Input variable
        - y: Output variable (quadratic function value)
        - task: Task identifier (source_a_b_c format or "target")
    """
    # Fixed parameters for data generation
    n_points = 100  # Dense grid for lookup coverage

    # Parameter sampling ranges
    a_range = (0.01, 2.0)  # Scale parameter
    b_range = (-1.0, 1.0) if not keep_min else (0.0, 0.0)  # Shift parameter
    c_range = (-2.0, 2.0)  # Offset parameter

    # Generate input points (same for all tasks) - dense grid for lookup
    x = np.linspace(-2, 2, n_points)

    all_data = []
    task_names = []

    # Generate source tasks
    for i in range(n_sources):
        # Sample parameters for source task
        a = np.random.uniform(*a_range)
        b = np.random.uniform(*b_range)
        c = np.random.uniform(*c_range)

        # Generate quadratic function: y = a*(x+b)^2 + c (no noise for lookup)
        y_clean = a * (x + b) ** 2 + c
        y_noisy = y_clean + np.random.normal(0, noise_std, n_points)

        # Create task name
        task_name = f"source_{i}"
        task_names.append(task_name)

        # Create DataFrame for this source
        source_df = pd.DataFrame({"x": x, "y": y_noisy, "task": task_name})
        all_data.append(source_df)

    # Generate target task
    a_target = np.random.uniform(*a_range)
    c_target = np.random.uniform(*c_range)
    # Fix c_target to 0 for convergence benchmarks
    b_target = 0

    y_target_clean = a_target * (x + b_target) ** 2 + c_target
    y_target_noisy = y_target_clean + np.random.normal(0, noise_std, n_points)

    target_df = pd.DataFrame({"x": x, "y": y_target_noisy, "task": "target"})
    all_data.append(target_df)
    task_names.append("target")

    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)

    return combined_data


def make_searchspace(
    data: pd.DataFrame,
    target_tasks: Sequence[str] | None = None,
    source_tasks: Sequence[str] | None = None,
) -> SearchSpace:
    """Create the search space for the benchmark.

    Args:
        data: DataFrame containing the quadratic data
        target_tasks: List of target task names (None for non-TL)
        source_tasks: List of source task names (None for non-TL)

    Returns:
        SearchSpace configured for the benchmark
    """
    # Extract discrete x values from data
    x_values = sorted(data["x"].unique())

    params: list[Parameter] = [NumericalDiscreteParameter("x", values=x_values)]

    # Add task parameter for transfer learning
    if target_tasks is not None and source_tasks is not None:
        all_tasks = [*source_tasks, *target_tasks]
        params.append(
            TaskParameter(
                name="task",
                values=all_tasks,
                active_values=target_tasks,
            )
        )

    return SearchSpace.from_product(parameters=params)


def make_lookup(data: pd.DataFrame, target_tasks: Sequence[str]) -> pd.DataFrame:
    """Create the lookup for the benchmark.

    Without the filtering, there would be multiple entries for the same parameter
    configuration. Since this might yield issues for the non-transfer learning
    campaigns, we filter the data to only include the target tasks.

    Args:
        data: DataFrame containing all task data
        target_tasks: List of target task names to include

    Returns:
        Filtered DataFrame containing only target task data
    """
    return data[data["task"].isin(target_tasks)]


def make_initial_data(data: pd.DataFrame, source_tasks: Sequence[str]) -> pd.DataFrame:
    """Create the initial data for the benchmark.

    Args:
        data: DataFrame containing all task data
        source_tasks: List of source task names to include

    Returns:
        Filtered DataFrame containing only source task data
    """
    return data[data["task"].isin(source_tasks)]


def quadratic_tl_convergence_benchmark(
    settings: ConvergenceBenchmarkSettings,
    source_tasks: Sequence[str],
    target_tasks: Sequence[str],
    percentages: Sequence[float],
    n_sources: int = 3,
    keep_min: bool = True,
) -> pd.DataFrame:
    """Abstract benchmark function comparing TL and non-TL campaigns on quadratics.

    This benchmark tests Bayesian optimization convergence with and without transfer
    learning on synthetic quadratic functions of the form y = a*(x+b)^2 + c.

    Args:
        settings: Configuration settings for the convergence benchmark
        source_tasks: List of source task identifiers
        target_tasks: List of target task identifiers
        percentages: Fractions of source data to use for initial data
        n_sources: Number of source tasks to generate
        keep_min: If True, all functions have same minimum location (b=0)

    Returns:
        DataFrame containing benchmark results comparing TL vs non-TL performance

    The benchmark structure follows:
    • Input: x (continuous, bounded by data range)
    • Task parameter: task (for TL campaigns)
    • Output: y (continuous, to be minimized)
    • Objective: Minimization of quadratic function
    • Comparison: Transfer learning vs. non-transfer learning campaigns
    """
    # Load data with specified configuration
    data, _ = load_data(n_sources=n_sources, keep_min=keep_min)

    # Create search spaces
    searchspace = make_searchspace(
        data=data,
        source_tasks=source_tasks,
        target_tasks=target_tasks,
    )
    searchspace_nontl = make_searchspace(data=data)

    # Prepare data splits
    lookup = make_lookup(data, target_tasks)
    initial_data = make_initial_data(data, source_tasks)

    # Define objective (minimize quadratic function)
    objective = SingleTargetObjective(NumericalTarget(name="y", mode="MIN"))

    # Create campaigns
    tl_campaign = Campaign(
        searchspace=searchspace,
        objective=objective,
    )
    nontl_campaign = Campaign(searchspace=searchspace_nontl, objective=objective)

    # Run benchmark scenarios with different percentages of source data
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
                    initial_data.sample(frac=p, random_state=42)
                    for _ in range(settings.n_mc_iterations)
                ],
                batch_size=settings.batch_size,
                n_doe_iterations=settings.n_doe_iterations,
                impute_mode="error",
            )
        )

    # Run baseline scenario with no initial data
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
