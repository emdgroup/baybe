"""Base functionality for quadratic transfer learning regression benchmarks.

This module provides common functionality for regression benchmarks using synthetic
quadratic functions: y = a*(x+b)^2 + c. Unlike convergence benchmarks, these focus
on predictive performance metrics (RMSE, R2, MAE, etc.) rather than optimization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalContinuousParameter, TaskParameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from benchmarks.definition.regression import TransferLearningRegressionSettings
from benchmarks.domains.regression.base import run_tl_regression_benchmark

# Define the benchmark settings
benchmark_config = TransferLearningRegressionSettings(
    random_seed=42,
    num_mc_iterations=30,
    max_train_points=5,
    source_fractions=[0.01, 0.02, 0.05, 0.10],
    noise_std=0.0,  # Noise is already added in data generation
    metrics=["RMSE", "R2", "MAE"],
)


def load_quadratic_data(n_sources: int = 3, keep_min: bool = False) -> pd.DataFrame:
    """Load synthetic quadratic data for transfer learning regression benchmarks.

    Creates source and target tasks using quadratic functions: y = a*(x+b)^2 + c
    with added noise for realistic regression scenarios.

    Args:
        n_sources: Number of source tasks to generate (default: 3)
        keep_min: If True, freeze b=0 for all functions (same minimum location)

    Returns:
        DataFrame containing both source and target task data with columns:
        - x: Input variable
        - y: Output variable (quadratic function value + noise)
        - task: Task identifier (source_a_b_c format or "target")
    """
    # Fixed parameters for data generation
    n_points = 100
    noise_std = 0.05
    seed = 42

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Parameter sampling ranges
    a_range = (0.1, 2.0)  # Scale parameter
    b_range = (-1.0, 1.0) if not keep_min else (0.0, 0.0)  # Shift parameter
    c_range = (-2.0, 2.0)  # Offset parameter

    # Generate input points (same for all tasks)
    x = np.linspace(-2, 2, n_points)

    all_data = []

    # Generate source tasks
    for i in range(n_sources):
        # Sample parameters for source task
        a = np.random.uniform(*a_range)
        b = np.random.uniform(*b_range)
        c = np.random.uniform(*c_range)

        # Generate quadratic function: y = a*(x+b)^2 + c
        y_clean = a * (x + b) ** 2 + c
        y_noisy = y_clean + np.random.normal(0, noise_std, n_points)

        # Create task name using integer ID for MHGP compatibility
        task_name = f"source_{i}"

        # Create DataFrame for this source
        source_df = pd.DataFrame({"x": x, "y": y_noisy, "task": task_name})
        all_data.append(source_df)

    # Generate target task
    a_target = np.random.uniform(*a_range)
    b_target = np.random.uniform(*b_range)
    c_target = np.random.uniform(*c_range)

    y_target_clean = a_target * (x + b_target) ** 2 + c_target
    y_target_noisy = y_target_clean + np.random.normal(0, noise_std, n_points)

    target_df = pd.DataFrame({"x": x, "y": y_target_noisy, "task": "target"})
    all_data.append(target_df)

    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)

    return combined_data


def create_quadratic_searchspaces(
    data: pd.DataFrame,
) -> tuple[SearchSpace, SearchSpace, str, list[str], str]:
    """Create search spaces for vanilla GP and transfer learning models.

    Args:
        data: DataFrame containing the quadratic data.

    Returns:
        Tuple containing:
        - vanilla_searchspace: SearchSpace for vanilla GP (no task parameter)
        - tl_searchspace: SearchSpace for transfer learning (with task parameter)
        - name_task: Name of the task parameter
        - source_tasks: List of source task values
        - target_task: Target task value
    """
    # Extract input bounds from data
    x_min = data["x"].min()
    x_max = data["x"].max()

    # Parameters for both search spaces
    params = [NumericalContinuousParameter("x", bounds=(x_min, x_max))]

    # Create vanilla GP search space (no task parameter)
    vanilla_searchspace = SearchSpace.from_product(params)

    # Extract task information from data
    all_tasks = data["task"].unique()
    target_task = "target"
    source_tasks = [task for task in all_tasks if task != target_task]

    # Create transfer learning search space (with task parameter)
    name_task = "task"
    task_param = TaskParameter(
        name=name_task,
        values=source_tasks + [target_task],
        active_values=[target_task],
    )

    tl_params = params + [task_param]
    tl_searchspace = SearchSpace.from_product(tl_params)

    return vanilla_searchspace, tl_searchspace, name_task, source_tasks, target_task


def create_quadratic_objective() -> SingleTargetObjective:
    """Create the objective for quadratic regression benchmarks."""
    return SingleTargetObjective(NumericalTarget(name="y", mode="MIN"))


def run_quadratic_tl_regression_benchmark(
    settings: TransferLearningRegressionSettings,
    n_sources: int = 3,
    keep_min: bool = False,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Run a quadratic transfer learning regression benchmark.

    Args:
        settings: The benchmark settings.
        n_sources: Number of source tasks to generate (default: 3)
        keep_min: If True, freeze b=0 for all functions (same minimum location)

    Returns:
        Tuple containing:
        - DataFrame with benchmark results
        - List of metric names used
        - List of model names used
    """
    return run_tl_regression_benchmark(
        settings=settings,
        load_data_fn=load_quadratic_data,
        create_searchspaces_fn=create_quadratic_searchspaces,
        create_objective_fn=create_quadratic_objective,
        load_data_kwargs={"n_sources": n_sources, "keep_min": keep_min},
    )
