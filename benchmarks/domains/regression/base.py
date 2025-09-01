"""Base implementation for transfer learning regression benchmarks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from tqdm import tqdm

from baybe.objectives import SingleTargetObjective
from baybe.parameters import TaskParameter
from baybe.searchspace import SearchSpace
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from benchmarks.definition import TransferLearningRegressionBenchmarkSettings


def kendall_tau_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Kendall's Tau correlation coefficient.

    Args:
        y_true: True target values
        y_pred: Predicted target values

    Returns:
        Kendall's Tau correlation coefficient
    """
    tau, _ = kendalltau(y_true, y_pred)
    return tau


def spearman_rho_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Spearman's Rho correlation coefficient.

    Args:
        y_true: True target values
        y_pred: Predicted target values

    Returns:
        Spearman's Rho correlation coefficient
    """
    rho, _ = spearmanr(y_true, y_pred)
    return rho


def run_tl_regression_benchmark(
    settings: TransferLearningRegressionBenchmarkSettings,
    load_data_fn: Callable[..., pd.DataFrame],
    make_searchspace_fn: Callable[[pd.DataFrame, bool], SearchSpace],
    create_objective_fn: Callable[..., SingleTargetObjective],
) -> pd.DataFrame:
    """Run a transfer learning regression benchmark.

    This function evaluates the performance of transfer learning models compared to
    non transfer learning models on regression tasks. It varies the fraction of
    source data used for training and the number of training points in the target task.

    For each combination, it trains a surrogate on target data only and transfer
    learning models on source + target data, then evaluates their predictive
    performance on held-out target test data using the provided regression metrics.

    Args:
        settings: The benchmark settings.
        load_data_fn: Function that loads the dataset.
        make_searchspace_fn: Function that creates search spaces for
            non-TL and TL models.
        create_objective_fn: Function that creates the objective function.

    Returns:
        DataFrame with benchmark results containing performance metrics for each
        model, training scenario, and Monte Carlo iteration.
    """
    objective = create_objective_fn()
    data = load_data_fn()

    # Create search space without task parameter
    vanilla_searchspace = make_searchspace_fn(data=data, use_task_parameter=False)

    # Create transfer learning search space (with task parameter)
    tl_searchspace = make_searchspace_fn(data=data, use_task_parameter=True)

    # Extract task parameter details
    task_param = next(
        p for p in tl_searchspace.parameters if isinstance(p, TaskParameter)
    )
    name_task = task_param.name

    # Extract single target task
    target_task = task_param.active_values[0]
    all_values = task_param.values
    source_tasks = [val for val in all_values if val != target_task]

    # Split data into source and target
    source_data = data[data[name_task].isin(source_tasks)]
    target_data = data[data[name_task] == target_task]

    # Collect all benchmark results across MC iterations and scenarios
    results: list[dict[str, Any]] = []

    # Create progress bar for Monte Carlo iterations
    mc_iter_bar = tqdm(
        range(settings.n_mc_iterations),
        desc="Monte Carlo iterations",
        unit="iter",
        position=0,
        leave=True,
    )

    for mc_iter in mc_iter_bar:
        # Create train/test split for target task
        target_indices = np.random.permutation(len(target_data))

        for fraction_source in settings.source_fractions:
            # Sample source data ensuring same fraction from each source task
            source_subset = _sample_source_data(
                source_data,
                source_tasks,
                fraction_source,
                name_task,
                settings.random_seed + mc_iter,
            )

            # Create progress bar for training points
            train_pts_bar = tqdm(
                range(1, settings.max_n_train_points + 1),
                desc=f"MC {mc_iter + 1}/{settings.n_mc_iterations},"
                f"Source frac {fraction_source:.1f}",
                unit="pts",
                position=1,
                leave=False,
            )

            for n_train_pts in train_pts_bar:
                train_indices = target_indices[:n_train_pts]
                test_indices = target_indices[
                    n_train_pts : n_train_pts + settings.max_n_train_points
                ]
                target_train = target_data.iloc[train_indices].copy()
                target_test = target_data.iloc[test_indices].copy()

                # Collect results for current training scenario
                scenario_results: list[dict[str, Any]] = []
                scenario_results.extend(
                    _evaluate_naive_models(
                        target_train,
                        target_test,
                        vanilla_searchspace,
                        tl_searchspace,
                        objective,
                        name_task,
                        target_task,
                    )
                )
                scenario_results.extend(
                    _evaluate_transfer_learning_models(
                        source_subset,
                        target_train,
                        target_test,
                        tl_searchspace,
                        objective,
                        fraction_source,
                        name_task,
                        target_task,
                    )
                )

                # Add metadata to each scenario result
                for scenario_result in scenario_results:
                    scenario_result.update(
                        {
                            "mc_iter": mc_iter,
                            "n_train_pts": n_train_pts,
                            "fraction_source": fraction_source,
                            "n_source_pts": len(source_subset),
                            "n_test_pts": len(target_test),
                            "source_data_seed": settings.random_seed + mc_iter,
                        }
                    )
                    results.append(scenario_result)

    results_df = pd.DataFrame(results)

    return results_df


def _create_tl_models() -> dict[str, GaussianProcessSurrogate]:
    """Create transfer learning model scenarios.

    Returns:
        Dictionary mapping model suffix names to initialized surrogate models
        for transfer learning evaluation.
    """
    return {
        "index_kernel": GaussianProcessSurrogate(),
    }


def _train_and_evaluate_model(
    model: GaussianProcessSurrogate,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    searchspace: SearchSpace,
    objective: SingleTargetObjective,
    scenario_name: str,
) -> dict[str, Any]:
    """Train a single model and evaluate its performance.

    Args:
        model: The surrogate model to train
        train_data: Training data
        test_data: Test data for evaluation
        searchspace: Search space for the model
        objective: Optimization objective
        scenario_name: Name of the scenario for results

    Returns:
        Dictionary with scenario name and evaluation metrics
    """
    target_column = objective._target.name
    train_data_prepared = train_data.copy()
    test_data_prepared = test_data.copy()

    model.fit(
        searchspace=searchspace, objective=objective, measurements=train_data_prepared
    )

    # Evaluate model
    predictions = model.posterior_stats(test_data_prepared, stats=["mean"])
    metrics = _calculate_metrics(
        true_values=test_data[target_column].values,
        predictions=predictions,
        target_column=target_column,
    )

    result = {"scenario": scenario_name}
    result.update(metrics)
    return result


def _sample_source_data(
    source_data: pd.DataFrame,
    source_tasks: list[str],
    fraction_source: float,
    task_column: str,
    source_data_seed: int,
) -> pd.DataFrame:
    """Sample source data ensuring same fraction from each source task.

    Args:
        source_data: DataFrame containing all source task data
        source_tasks: List of source task identifiers
        fraction_source: Fraction of data to sample from each source task
        task_column: Name of column containing task identifiers
        source_data_seed: Random seed for reproducible sampling

    Returns:
        Combined DataFrame with sampled data from all source tasks
    """
    # Collect sampled subsets from each source task
    source_subsets: list[pd.DataFrame] = []

    for source_task in source_tasks:
        task_data = source_data[source_data[task_column] == source_task]
        if len(task_data) > 0:
            task_subset = task_data.sample(
                frac=fraction_source,
                random_state=source_data_seed,
            )
            source_subsets.append(task_subset)
    return pd.concat(source_subsets, ignore_index=True)


def _evaluate_naive_models(
    target_train: pd.DataFrame,
    target_test: pd.DataFrame,
    vanilla_searchspace: SearchSpace,
    tl_searchspace: SearchSpace,
    objective: SingleTargetObjective,
    task_column: str,
    task_value: str,
) -> list[dict[str, Any]]:
    """Evaluate both naive model baselines that do not use source data.

    Args:
        target_train: Target task training data
        target_test: Target task test data
        vanilla_searchspace: Search space without task parameter
        tl_searchspace: Search space with task parameter
        objective: Optimization objective
        task_column: Name of task parameter column
        task_value: Value for task parameter

    Returns:
        List of evaluation results for naive baselines
    """
    # Collect evaluation results for models without source data
    results: list[dict[str, Any]] = []

    # Naive GP on reduced search space (no source data, no task parameter)
    results.append(
        _train_and_evaluate_model(
            GaussianProcessSurrogate(),
            target_train,
            target_test,
            vanilla_searchspace,
            objective,
            "0_reduced_searchspace",
        )
    )

    # Naive GP on full search space (no source data, with task parameter)
    results.append(
        _train_and_evaluate_model(
            GaussianProcessSurrogate(),
            target_train,
            target_test,
            tl_searchspace,
            objective,
            "0_full_searchspace",
        )
    )

    return results


def _evaluate_transfer_learning_models(
    source_data: pd.DataFrame,
    target_train: pd.DataFrame,
    target_test: pd.DataFrame,
    tl_searchspace: SearchSpace,
    objective: SingleTargetObjective,
    fraction_source: float,
    task_column: str,
    task_value: str,
) -> list[dict[str, Any]]:
    """Evaluate all transfer learning models using source and target data.

    Args:
        source_data: Source task data
        target_train: Target task training data
        target_test: Target task test data
        tl_searchspace: Search space with task parameter
        objective: Optimization objective
        fraction_source: Fraction of source data used
        task_column: Name of task parameter column
        task_value: Value for task parameter

    Returns:
        List of evaluation results for transfer learning models
    """
    # Collect evaluation results for transfer learning models
    results: list[dict[str, Any]] = []
    combined_data = pd.concat([source_data, target_train])

    for model_suffix, model in _create_tl_models().items():
        scenario_name = f"{int(100 * fraction_source)}_{model_suffix}"
        results.append(
            _train_and_evaluate_model(
                model,
                combined_data,
                target_test,
                tl_searchspace,
                objective,
                scenario_name,
            )
        )

    return results


def _calculate_metrics(
    true_values: np.ndarray,
    predictions: pd.DataFrame,
    target_column: str,
) -> dict[str, float]:
    """Calculate regression metrics for model predictions.

    Args:
        true_values: True target values
        predictions: Model predictions DataFrame with mean columns
        target_column: Name of the target column

    Returns:
        Dictionary with metric names as keys and metric values as values
    """
    regression_metrics = [
        root_mean_squared_error,
        mean_squared_error,
        r2_score,
        mean_absolute_error,
        max_error,
        explained_variance_score,
        kendall_tau_score,
        spearman_rho_score,
    ]
    results = {}
    pred_values = predictions[f"{target_column}_mean"].values

    for metric_func in regression_metrics:
        metric_value = metric_func(true_values, pred_values)
        results[metric_func.__name__] = metric_value

    return results
