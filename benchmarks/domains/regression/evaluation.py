"""Common evaluation utilities for transfer learning benchmarks."""

from collections.abc import Sequence
from typing import Any

import pandas as pd

from baybe.objectives import SingleTargetObjective
from baybe.searchspace import SearchSpace
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from benchmarks.definition.regression import REGRESSION_METRICS


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
