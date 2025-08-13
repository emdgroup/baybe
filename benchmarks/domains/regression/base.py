"""Base implementation for transfer learning regression benchmarks."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from baybe.objectives import SingleTargetObjective
from baybe.searchspace import SearchSpace
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from benchmarks.definition import TransferLearningRegressionSettings
from benchmarks.definition.regression import REGRESSION_METRICS


def generate_result_filename(benchmark_name: str, extension: str = "csv") -> str:
    """Generate a consistent filename pattern for benchmark results.

    Args:
        benchmark_name: Name of the benchmark (e.g., "quadratic_tl_regression")
        extension: File extension (default: "csv")

    Returns:
        Filename following pattern:
        {benchmark-name}_{branch-info}_{version}_{date}_{commit-hash}_results.{extension}
    """
    try:
        # Get current branch
        branch_result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            check=True,
        )
        branch = branch_result.stdout.strip()
        branch = branch.replace("/", "-")  # Replace slashes to avoid directory issues
        print(f"Current branch: {branch}")

        # Get current commit hash
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        commit = commit_result.stdout.strip()

        # Get BayBE version (try to read from pyproject.toml or use fallback)
        try:
            import baybe

            version = baybe.__version__
        except (ImportError, AttributeError):
            version = "0.13.2"  # Fallback version

        # Get current date
        date = datetime.now().strftime("%Y-%m-%d")

        # Generate filename
        filename = (
            f"{benchmark_name}_{branch}_{version}_{date}_{commit}_results.{extension}"
        )

    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to simple naming if git is not available
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{benchmark_name}_results_{timestamp}.{extension}"

    return filename


def run_tl_regression_benchmark(
    settings: TransferLearningRegressionSettings,
    load_data_fn: Callable[..., pd.DataFrame],
    create_searchspaces_fn: Callable[
        [pd.DataFrame], tuple[SearchSpace, SearchSpace, str, list[str], str]
    ],
    create_objective_fn: Callable[[], SingleTargetObjective],
    load_data_kwargs: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Run a transfer learning regression benchmark.

    Args:
        settings: The benchmark settings.
        load_data_fn: Function that loads the dataset.
        create_searchspaces_fn: Function that creates search spaces.
        create_objective_fn: Function that creates the objective function.
        load_data_kwargs: Additional keyword arguments for load_data_fn.

    Returns:
        Tuple containing:
        - DataFrame with benchmark results
        - List of metric names used
        - List of model names used
    """
    # Set random seed for reproducibility
    np.random.seed(settings.random_seed)
    torch.manual_seed(settings.random_seed)

    # Create target objective
    objective = create_objective_fn()
    target_column = objective._target.name

    # Load data and create search spaces
    if load_data_kwargs is None:
        load_data_kwargs = {}
    data = load_data_fn(**load_data_kwargs)
    vanilla_searchspace, tl_searchspace, name_task, source_tasks, target_task = (
        create_searchspaces_fn(data)
    )

    # Split data into source and target
    source_data = data[data[name_task].isin(source_tasks)]
    target_data = data[data[name_task] == target_task]

    # Main benchmark loop
    results = []

    # Create progress bar for Monte Carlo iterations
    mc_iter_bar = tqdm(
        range(settings.num_mc_iterations),
        desc="Monte Carlo iterations",
        unit="iter",
        position=0,
        leave=True,
    )

    # Calculate total number of evaluations for overall progress
    total_evals = (
        settings.num_mc_iterations
        * len(settings.source_fractions)
        * settings.max_train_points
    )
    overall_progress = tqdm(
        total=total_evals, desc="Overall progress", unit="eval", position=1, leave=True
    )

    for mc_iter in mc_iter_bar:  # range(settings.num_mc_iterations)
        print(f"Monte Carlo iteration {mc_iter + 1}/{settings.num_mc_iterations}")

        # Create train/test split for target task
        target_indices = np.random.permutation(len(target_data))
        max_train_points = min(
            settings.max_train_points,
            len(target_data) - 10,  # Ensure at least 10 test points
        )

        # Create progress bar for source fractions
        source_fraction_bar = tqdm(
            settings.source_fractions,
            desc=f"MC iter {mc_iter + 1}/{settings.num_mc_iterations}: Source frac",
            unit="frac",
            position=2,
            leave=False,
        )

        for fraction_source in source_fraction_bar:  # settings.source_fractions:
            # Sample source data ensuring same fraction from each source task
            source_subsets = []
            for source_task in source_tasks:
                task_data = source_data[source_data[name_task] == source_task]
                if len(task_data) > 0:
                    task_subset = task_data.sample(
                        frac=fraction_source,
                        random_state=settings.random_seed + mc_iter,
                    )
                    source_subsets.append(task_subset)

            # Combine all source task subsets
            source_subset = pd.concat(source_subsets, ignore_index=True)

            # Create progress bar for training points
            train_pts_bar = tqdm(
                range(1, max_train_points + 1),
                desc=f"Source fraction {fraction_source:.2f}: Training points",
                unit="pts",
                position=3,
                leave=False,
            )

            # Generate the source data subset
            for n_train_pts in train_pts_bar:  # range(1, max_train_points + 1):
                # Create models
                vanilla_gp = GaussianProcessSurrogate()
                tl_models = [
                    {"name": "GP_Index_Kernel", "model": GaussianProcessSurrogate()},
                ]
                train_indices = target_indices[:n_train_pts]
                test_indices = target_indices[
                    n_train_pts : n_train_pts + 50
                ]  # Use up to 50 test points
                target_train = target_data.iloc[train_indices].copy()
                target_test = target_data.iloc[test_indices].copy()

                # Update progress bar description with current evaluation details
                train_pts_bar.set_description(
                    f"Source: {len(source_subset)} pts,"
                    f"Target train: {n_train_pts} pts,"
                    f"Test: {len(target_test)} pts"
                )

                # Evaluate models
                eval_results = evaluate_models(
                    vanilla_gp=vanilla_gp,
                    tl_models=tl_models,
                    source_data=source_subset,
                    target_train=target_train,
                    target_test=target_test,
                    vanilla_searchspace=vanilla_searchspace,
                    tl_searchspace=tl_searchspace,
                    objective=objective,
                    metrics=settings.metrics,
                    target_column=target_column,
                    task_column=name_task,
                    task_value=target_task,
                )

                # Add metadata
                eval_results.update(
                    {
                        "mc_iter": mc_iter,
                        "n_train_pts": n_train_pts,
                        "fraction_source": fraction_source,
                        "n_source_pts": len(source_subset),
                        "n_test_pts": len(target_test),
                    }
                )
                results.append(eval_results)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    overall_progress.close()

    # Return results
    return results_df


def evaluate_models(
    vanilla_gp: GaussianProcessSurrogate,
    tl_models: list[dict[str, Any]],
    source_data: pd.DataFrame,
    target_train: pd.DataFrame,
    target_test: pd.DataFrame,
    vanilla_searchspace: SearchSpace,
    tl_searchspace: SearchSpace,
    objective: SingleTargetObjective,
    metrics: Sequence[str],
    target_column: str = "y",
    task_column: str | None = None,
    task_value: str | None = None,
    drop_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Train models and evaluate their performance using specified metrics.

    Args:
        vanilla_gp: Vanilla GP model
        tl_models: List of transfer learning models with their names
        source_data: Source task data
        target_train: Target task training data
        target_test: Target task test data
        vanilla_searchspace: Search space for vanilla GP
        tl_searchspace: Search space for transfer learning models
        objective: Objective function
        metrics: Metrics to evaluate
        target_column: Name of the target column in the data
        task_column: Name of the task column (e.g., "task" or "Temp_C")
        task_value: Value to set for the task column in test data
        drop_columns: Columns to drop when making predictions

    Returns:
        Dictionary with evaluation results
    """
    # Define which stats to request from posterior_stats
    stats_to_request = ["mean"]

    # Results dictionary
    results = {}

    # Train vanilla GP on target data only
    vanilla_gp.fit(
        searchspace=vanilla_searchspace,
        objective=objective,
        measurements=target_train,
    )

    # Prepare test data for vanilla GP
    test_for_vanilla = target_test.copy()
    if drop_columns:
        test_for_vanilla = test_for_vanilla.drop(columns=drop_columns)

    # Evaluate vanilla GP
    vanilla_pred = vanilla_gp.posterior_stats(test_for_vanilla, stats=stats_to_request)

    # Calculate all requested metrics for vanilla GP
    for metric_name in metrics:
        metric_info = REGRESSION_METRICS[metric_name]
        metric_func = metric_info["function"]
        transform_func = metric_info["transform"]

        # Call with just mean predictions
        metric_value = metric_func(
            target_test[target_column].values,
            vanilla_pred[f"{target_column}_mean"].values,
        )

        # Apply transformation if needed (e.g., sqrt for RMSE)
        if transform_func is not None:
            metric_value = transform_func(metric_value)

        results[f"vanilla_{metric_name.lower()}"] = metric_value

    # Sample source data and prepare combined data for TL models
    combined_data = pd.concat([source_data, target_train])

    # Train and evaluate each TL model
    for tl_model_dict in tl_models:
        model_name = tl_model_dict["name"]
        tl_model = tl_model_dict["model"]

        # Train TL model with combined data
        tl_model.fit(
            searchspace=tl_searchspace,
            objective=objective,
            measurements=combined_data,
        )

        # Prepare test data for TL model
        test_data_with_task = target_test.copy()
        if task_column and task_value:
            test_data_with_task[task_column] = task_value

        # Drop columns if needed
        test_for_tl = test_data_with_task
        if drop_columns:
            drop_cols = [col for col in drop_columns if col != task_column]
            test_for_tl = test_data_with_task.drop(columns=drop_cols)

        # Evaluate TL model
        tl_pred = tl_model.posterior_stats(test_for_tl, stats=stats_to_request)

        # Calculate all requested metrics
        for metric_name in metrics:
            metric_info = REGRESSION_METRICS[metric_name]
            metric_func = metric_info["function"]
            transform_func = metric_info["transform"]

            # Call with just mean predictions
            metric_value = metric_func(
                target_test[target_column].values,
                tl_pred[f"{target_column}_mean"].values,
            )

            # Apply transformation if needed (e.g., sqrt for RMSE)
            if transform_func is not None:
                metric_value = transform_func(metric_value)

            results[f"{model_name}_{metric_name.lower()}"] = metric_value

    return results
