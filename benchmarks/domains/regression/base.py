"""Base implementation for transfer learning regression benchmarks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch

from baybe.objectives import SingleTargetObjective
from baybe.searchspace import SearchSpace
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.surrogates.transfergpbo import MHGPGaussianProcessSurrogate
from benchmarks.definition import TransferLearningRegressionSettings
from benchmarks.definition.regression import REGRESSION_METRICS


def run_tl_regression_benchmark(
    settings: TransferLearningRegressionSettings,
    load_data_fn: Callable[[], pd.DataFrame],
    create_searchspaces_fn: Callable[
        [pd.DataFrame], tuple[SearchSpace, SearchSpace, str, list[str], str]
    ],
    create_objective_fn: Callable[[], SingleTargetObjective],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Run a transfer learning regression benchmark.

    Args:
        settings: The benchmark settings.
        load_data_fn: Function that loads the dataset.
        create_searchspaces_fn: Function that creates search spaces.
        create_objective_fn: Function that creates the objective function.

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
    data = load_data_fn()
    vanilla_searchspace, tl_searchspace, name_task, source_tasks, target_task = (
        create_searchspaces_fn(data)
    )

    # Split data into source and target
    source_data = data[data[name_task].isin(source_tasks)]
    target_data = data[data[name_task] == target_task]

    # Main benchmark loop
    results = []
    for mc_iter in range(settings.num_mc_iterations):
        print(f"Monte Carlo iteration {mc_iter + 1}/{settings.num_mc_iterations}")

        # Create train/test split for target task
        target_indices = np.random.permutation(len(target_data))
        max_train_points = min(
            settings.max_train_points,
            len(target_data) - 10,  # Ensure at least 10 test points
        )

        # Create models
        vanilla_gp = GaussianProcessSurrogate()
        tl_models = [
            {
                "name": "MHGP_Stable",
                "model": MHGPGaussianProcessSurrogate(numerical_stability=True),
            },
            {"name": "GP_Index_Kernel", "model": GaussianProcessSurrogate()},
        ]

        for fraction_source in settings.source_fractions:
            # Sample source data and keep it constant for all models
            source_subset = source_data.sample(
                frac=fraction_source, random_state=settings.random_seed + mc_iter
            )

            # Generate the source data subset
            for n_train_pts in range(1, max_train_points + 1):
                train_indices = target_indices[:n_train_pts]
                test_indices = target_indices[
                    n_train_pts : n_train_pts + 50
                ]  # Use up to 50 test points
                target_train = target_data.iloc[train_indices].copy()
                target_test = target_data.iloc[test_indices].copy()

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

    # Extract model names for return value
    model_names = ["vanilla"] + [model_dict["name"] for model_dict in tl_models]

    # Return results, metrics, and model names for plotting
    return results_df, settings.metrics, model_names


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
    # Determine if we need variance for any of the requested metrics
    need_variance = any(
        REGRESSION_METRICS[metric]["needs_variance"] for metric in metrics
    )

    # Define which stats to request from posterior_stats
    stats_to_request = ["mean"]
    if need_variance:
        stats_to_request.append("var")

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

        if metric_info["needs_variance"]:
            # Call with both mean and variance
            metric_value = metric_func(
                y_true=target_test[target_column].values,
                y_pred_mean=vanilla_pred[f"{target_column}_mean"].values,
                y_pred_var=vanilla_pred[f"{target_column}_var"].values,
            )
        else:
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

            if metric_info["needs_variance"]:
                # Call with both mean and variance
                metric_value = metric_func(
                    y_true=target_test[target_column].values,
                    y_pred_mean=tl_pred[f"{target_column}_mean"].values,
                    y_pred_var=tl_pred[f"{target_column}_var"].values,
                )
            else:
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
