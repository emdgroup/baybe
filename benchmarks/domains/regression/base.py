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

from baybe.parameters import TaskParameter
from baybe.objectives import SingleTargetObjective
from baybe.searchspace import SearchSpace
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
# from baybe.surrogates.source_prior import SourcePriorGaussianProcessSurrogate
# from baybe.surrogates.transfergpbo import (
#     MHGPGaussianProcessSurrogate,
#     SHGPGaussianProcessSurrogate,
# )
from benchmarks.definition import TransferLearningRegressionBenchmarkSettings


def kendall_tau_score(y_true, y_pred):
    """Calculate Kendall's Tau correlation coefficient."""
    tau, _ = kendalltau(y_true, y_pred)
    return tau


def spearman_rho_score(y_true, y_pred):
    """Calculate Spearman's Rho correlation coefficient."""
    rho, _ = spearmanr(y_true, y_pred)
    return rho


def run_tl_regression_benchmark(
    settings: TransferLearningRegressionBenchmarkSettings,
    load_data_fn: Callable[..., pd.DataFrame],
    make_searchspace_fn: Callable[
        [pd.DataFrame, bool], SearchSpace],
    create_objective_fn: Callable[[], SingleTargetObjective],
    load_data_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run a transfer learning regression benchmark.

    This function evaluates the performance of transfer learning models compared to
    vanilla Gaussian Process models on regression tasks. It varies the fraction of
    source data used for training and the number of training points in the target task.

    For each combination, it trains both vanilla GP (target data only) and transfer
    learning models (source + target data), then evaluates their predictive
    performance on held-out target test data using the provided regression metrics.

    Args:
        settings: The benchmark settings.
        load_data_fn: Function that loads the dataset.
        create_searchspaces_fn: Function that creates search spaces for
            non-TL and TL models, name of the task parameter, list
            of source task names and name of the target task.
        create_objective_fn: Function that creates the objective function.
        load_data_kwargs: Additional keyword arguments for load_data_fn.

    Returns:
        DataFrame with benchmark results containing performance metrics for each
        model, training scenario, and Monte Carlo iteration.
    """
    # Create target objective
    objective = create_objective_fn()
    target_column = objective._target.name

    # Load data and create search spaces
    if load_data_kwargs is None:
        load_data_kwargs = {}
    data = load_data_fn(**load_data_kwargs)
    # Create SearchSpace without task parameter (vanilla GP)
    vanilla_searchspace = make_searchspace_fn(data=data, use_task_parameter=False)

    # Create transfer learning search space (with task parameter)
    tl_searchspace = make_searchspace_fn(data=data, use_task_parameter=True)

    # Extract task parameter details
    task_param = next(
        p for p in tl_searchspace.parameters if isinstance(p, TaskParameter)
    )
    name_task = task_param.name
    target_task = task_param.active_values[0]  # Extract single target task
    all_values = task_param.values
    source_tasks = [val for val in all_values if val != target_task]

    # Split data into source and target
    source_data = data[data[name_task].isin(source_tasks)]
    target_data = data[data[name_task] == target_task]

    # Main benchmark loop
    results = []

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
                  source_data, source_tasks, fraction_source,
                  name_task, settings.random_seed, mc_iter
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

                # Evaluate models - now returns list of scenario results
                scenario_results = _evaluate_models(
                    fraction_source=fraction_source,
                    source_data=source_subset,
                    target_train=target_train,
                    target_test=target_test,
                    vanilla_searchspace=vanilla_searchspace,
                    tl_searchspace=tl_searchspace,
                    objective=objective,
                    target_column=target_column,
                    task_column=name_task,
                    task_value=target_task,
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
                        }
                    )
                    results.append(scenario_result)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    return results_df

def _create_tl_models() -> dict[str, GaussianProcessSurrogate]:
    """Create transfer learning model scenarios.


    Returns:
        Dictionary with models to use for TL."""
    return {
        "index_kernel": GaussianProcessSurrogate(),
        # "mhgp": MHGPGaussianProcessSurrogate(),
        # "shgp": SHGPGaussianProcessSurrogate(),
        # "source_prior": SourcePriorGaussianProcessSurrogate(),
    }

def _sample_source_data(
    source_data: pd.DataFrame, 
    source_tasks: list[str], 
    fraction_source: float,
    task_column: str,
    random_seed: int,
    mc_iter: int
) -> pd.DataFrame:
    """Sample source data ensuring same fraction from each source task."""
    source_subsets = []
    for source_task in source_tasks:
        task_data = source_data[source_data[task_column] == source_task]
        if len(task_data) > 0:
            task_subset = task_data.sample(
                frac=fraction_source,
                random_state=random_seed + mc_iter,
            )
            source_subsets.append(task_subset)
    return pd.concat(source_subsets, ignore_index=True)

def _evaluate_models(
    fraction_source: float,
    source_data: pd.DataFrame,
    target_train: pd.DataFrame,
    target_test: pd.DataFrame,
    vanilla_searchspace: SearchSpace,
    tl_searchspace: SearchSpace,
    objective: SingleTargetObjective,
    target_column: str = "y",
    task_column: str | None = None,
    task_value: str | None = None,
) -> list[dict[str, Any]]:
    """Train models and evaluate their performance using specified metrics.

    Args:
        fraction_source: Fraction of source data
        source_data: Source task data
        target_train: Target task training data
        target_test: Target task test data
        vanilla_searchspace: Search space for vanilla GP
        tl_searchspace: Search space for transfer learning models
        objective: Objective function
        target_column: Name of the target column in the data
        task_column: Name of the task column (e.g., "task" or "Temp_C")
        task_value: Value to set for the task column in test data

    Returns:
        List of dictionaries, one per scenario with scenario column and metrics
    """
    # Crurrent implemented metrics require mean predictions only
    stats_to_request = ["mean"]

    # Results list - one entry per scenario
    results = []

    # Create models
    gp_reduced = GaussianProcessSurrogate()

    # Train GP on the reduced searchspace and the target data only
    gp_reduced.fit(
        searchspace=vanilla_searchspace,
        objective=objective,
        measurements=target_train,
    )

    # Prepare test data for GP on the reduced searchspace
    test_for_gp_reduced = target_test.copy()

    # Evaluate GP on the reduced searchspace
    gp_reduced_pred = gp_reduced.posterior_stats(test_for_gp_reduced, stats=stats_to_request)

    # Calculate all requested metrics for GP on the reduced searchspace
    gp_reduced_metrics = _calculate_metrics(
        true_values=target_test[target_column].values,
        predictions=gp_reduced_pred,
        target_column=target_column,
    )
    
    gp_reduced_result = {"scenario": "0_reduced_searchspace"}
    gp_reduced_result.update(gp_reduced_metrics)
    results.append(gp_reduced_result)

    # Add the GP on full searchspace baseline (matches "0" scenario from ConvergenceBenchmark)
    gp_full = GaussianProcessSurrogate()
    
    # Prepare training data with task parameter for GP on full searchspace
    target_train_with_task = target_train.copy()
    if task_column and task_value:
        target_train_with_task[task_column] = task_value
    
    # Train GP on the full searchspace and the target data only
    gp_full.fit(
        searchspace=tl_searchspace,
        objective=objective,
        measurements=target_train_with_task,
    )
    
    # Prepare test data for GP on full searchspace
    test_for_gp_full = target_test.copy()
    if task_column and task_value:
        test_for_gp_full[task_column] = task_value

    # Evaluate GP on full search space
    gp_full_pred = gp_full.posterior_stats(test_for_gp_full, stats=stats_to_request)

    # Calculate all requested metrics for GP on full searchspace
    gp_full_metrics = _calculate_metrics(
        true_values=target_test[target_column].values,
        predictions=gp_full_pred,
        target_column=target_column,
    )
    
    gp_full_result = {"scenario": "0_full_searchspace"}
    gp_full_result.update(gp_full_metrics)
    results.append(gp_full_result)

    tl_scenarios = _create_tl_models()

    # Sample source data and prepare combined data for TL models
    combined_data = pd.concat([source_data, target_train])

    # Train and evaluate each TL model
    for model_suffix, tl_model in tl_scenarios.items():
        scenario = f"{int(100 * fraction_source)}_" + model_suffix
        # Train TL model with combined data
        tl_model.fit(
            searchspace=tl_searchspace,
            objective=objective,
            measurements=combined_data,
        )

        # Prepare test data for TL model
        test_for_tl = target_test.copy()
        if task_column and task_value:
            test_for_tl[task_column] = task_value

        # Evaluate TL model
        tl_pred = tl_model.posterior_stats(test_for_tl, stats=stats_to_request)

        # Calculate all requested metrics
        tl_metrics = _calculate_metrics(
            true_values=target_test[target_column].values,
            predictions=tl_pred,
            target_column=target_column,
        )
        
        # Create TL model result row
        tl_result = {"scenario": scenario}
        tl_result.update(tl_metrics)
        results.append(tl_result)

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
        Dictionary with metric results
    """
    # List of regression metric functions
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