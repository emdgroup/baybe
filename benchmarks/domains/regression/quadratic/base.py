"""Base functionality for quadratic transfer learning regression benchmarks.

This module provides common functionality for regression benchmarks using synthetic
quadratic functions: y = a*(x+b)^2 + c. Unlike convergence benchmarks, these focus
on predictive performance metrics (RMSE, R2, MAE, etc.) rather than optimization.
"""

from __future__ import annotations

import pandas as pd

from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalContinuousParameter, TaskParameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from benchmarks.definition.regression import TransferLearningRegressionSettings
from benchmarks.domains.regression.base import run_tl_regression_benchmark
from benchmarks.domains.transfer_learning.quadratic.base import load_data

# Define the benchmark settings
benchmark_config = TransferLearningRegressionSettings(
    random_seed=42,
    num_mc_iterations=2,  # 30,
    max_train_points=1,  # 5,
    source_fractions=[0.01],  # , 0.02, 0.05, 0.10],
    noise_std=0.0,  # Noise is already added in data generation
    metrics=["RMSE", "R2", "MAE"],
)


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
) -> pd.DataFrame:
    """Run a quadratic transfer learning regression benchmark.

    Args:
        settings: The benchmark settings.
        n_sources: Number of source tasks to generate (default: 3)
        keep_min: If True, freeze b=0 for all functions (same minimum location)

    Returns:
        DataFrame with benchmark results
    """
    return run_tl_regression_benchmark(
        settings=settings,
        load_data_fn=load_data,
        create_searchspaces_fn=create_quadratic_searchspaces,
        create_objective_fn=create_quadratic_objective,
        load_data_kwargs={"n_sources": n_sources, "keep_min": keep_min},
    )
