"""Base implementation for transfer learning regression benchmarks."""

from __future__ import annotations

from typing import Any, Protocol

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
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from baybe.objectives import SingleTargetObjective
from baybe.parameters import TaskParameter
from baybe.searchspace import SearchSpace
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from benchmarks.definition import TransferLearningRegressionBenchmarkSettings


class DataLoader(Protocol):
    """Protocol for data loading functions used in TL regression benchmarks."""

    def __call__(self) -> pd.DataFrame:
        """Load and return the dataset for regression benchmark evaluation.

        Returns:
            DataFrame containing the data with parameters and target values.
        """
        ...


class SearchSpaceFactory(Protocol):
    """Protocol for SearchSpace creation used in TL regression benchmarks."""

    def __call__(self, data: pd.DataFrame, use_task_parameter: bool) -> SearchSpace:
        """Create a SearchSpace for regression benchmark evaluation.

        Args:
            data: The dataset to create the search space from.
            use_task_parameter: Whether to include task parameter for TL
                scenarios. If True, creates search space with TaskParameter for
                TL models. If False, creates vanilla search space without
                task parameter.

        Returns:
            The TL and non-TL searchspaces for the benchmark.
        """
        ...


class ObjectiveFactory(Protocol):
    """Protocol for objective creation functions used in regression benchmarks."""

    def __call__(self) -> SingleTargetObjective:
        """Create and return the optimization objective for regression benchmarks.

        Returns:
            The objective of the benchmark.
        """
        ...


def kendall_tau_score(x: np.ndarray, y: np.ndarray, /) -> float:
    """Calculate Kendall's Tau correlation coefficient.

    Values close to 1 indicate strong positive correlation, values close to -1
    indicate strong negative correlation, and values near 0 indicate no
    correlation.

    Args:
        x: First array of values.
        y: Second array of values, same shape as x.

    Returns:
        Kendall's Tau correlation coefficient.
    """
    tau, _ = kendalltau(x, y)
    return tau


def spearman_rho_score(x: np.ndarray, y: np.ndarray, /) -> float:
    """Calculate Spearman's Rho correlation coefficient.

    Values close to 1 indicate strong positive monotonic correlation,
    values close to -1 indicate strong negative monotonic correlation.

    Args:
        x: First array of values.
        y: Second array of values, same shape as x.

    Returns:
        Spearman's Rho correlation coefficient.
    """
    rho, _ = spearmanr(x, y)
    return rho


# Dictionary mapping transfer learning model names to their surrogate classes
TL_MODELS = {
    "index_kernel": GaussianProcessSurrogate,
}


# List of regression metrics to evaluate model performance
REGRESSION_METRICS = {
    root_mean_squared_error,
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    max_error,
    explained_variance_score,
    kendall_tau_score,
    spearman_rho_score,
}


def run_tl_regression_benchmark(
    settings: TransferLearningRegressionBenchmarkSettings,
    data_loader: DataLoader,
    searchspace_factory: SearchSpaceFactory,
    objective_factory: ObjectiveFactory,
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
        data_loader: Function that loads the dataset for regression evaluation.
        searchspace_factory: Function that creates search spaces for both
            non-TL and TL model scenarios.
        objective_factory: Function that creates the optimization objective.

    Returns:
        DataFrame with benchmark results containing performance metrics for each
        model, training scenario, and Monte Carlo iteration. Columns include:

      - scenario: Model scenario identifier (e.g., "0_reduced_searchspace",
        "5_index_kernel")
      - Performance metrics: root_mean_squared_error, mean_squared_error, r2_score,
        mean_absolute_error, max_error, explained_variance_score, kendall_tau_score,
        spearman_rho_score
      - Experimental metadata: mc_iter, n_train_pts, fraction_source, n_source_pts,
        n_test_pts, source_data_seed

    """
    objective = objective_factory()
    data = data_loader()

    # Create search space without task parameter
    vanilla_searchspace = searchspace_factory(data=data, use_task_parameter=False)

    # Create transfer learning search space (with task parameter)
    tl_searchspace = searchspace_factory(data=data, use_task_parameter=True)

    # Extract task parameter details
    task_param = next(
        p for p in tl_searchspace.parameters if isinstance(p, TaskParameter)
    )
    name_task = task_param.name

    # Extract target tasks (can be multiple)
    target_tasks = task_param.active_values
    source_tasks = [val for val in task_param.values if val not in target_tasks]

    # Split data into source and target
    source_data = data[data[name_task].isin(source_tasks)]
    target_data = data[data[name_task].isin(target_tasks)]

    # Ensure sufficient target data for train/test splits
    assert len(target_data) >= 2 * settings.max_n_train_points, (
        f"Insufficient target data:"
        f"{len(target_data)} < {2 * settings.max_n_train_points}"
    )

    # Collect all benchmark results across MC iterations and scenarios
    results: list[dict[str, Any]] = []

    # Calculate total iterations for single progress bar
    total_iterations = (
        settings.n_mc_iterations
        * len(settings.source_fractions)
        * settings.max_n_train_points
    )

    # Single progress bar for all iterations
    pbar = tqdm(total=total_iterations, desc="Running benchmark", unit="eval")

    for mc_iter in range(settings.n_mc_iterations):
        for n_train_pts in range(1, settings.max_n_train_points + 1):
            target_train, target_test = train_test_split(
                target_data,
                train_size=n_train_pts,
                test_size=settings.max_n_train_points,
                random_state=settings.random_seed + mc_iter,
                shuffle=True,
            )

            # Add noise to target training data
            if settings.noise_std > 0:
                target_train = target_train.copy()
                target_train[objective._target.name] += np.random.normal(
                    0, settings.noise_std, len(target_train)
                )

            # Evaluate naive models once per (mc_iter, n_train_pts)
            naive_results = _evaluate_naive_models(
                target_train,
                target_test,
                vanilla_searchspace,
                tl_searchspace,
                objective,
            )

            # Add metadata for naive models (fraction_source = 0 for consistency)
            for naive_result in naive_results:
                naive_result.update(
                    {
                        "mc_iter": mc_iter,
                        "n_train_pts": n_train_pts,
                        "fraction_source": 0.0,  # Naive models don't use source data
                        "n_source_pts": 0,
                        "n_test_pts": len(target_test),
                        "source_data_seed": settings.random_seed + mc_iter,
                    }
                )
                results.append(naive_result)

            # Evaluate transfer learning models for each source fraction
            for fraction_source in settings.source_fractions:
                # Update progress bar description
                pbar.set_description(
                    f"MC {mc_iter + 1}/{settings.n_mc_iterations} | "
                    f"Frac {fraction_source:.2f} | "
                    f"Pts {n_train_pts}/{settings.max_n_train_points}"
                )

                # Sample source data using configured sampling strategy
                source_subset = _sample_source_data(
                    source_data,
                    source_tasks,
                    fraction_source,
                    name_task,
                    settings.random_seed + mc_iter,
                    settings.stratified_source_sampling,
                )

                tl_results = _evaluate_transfer_learning_models(
                    source_subset,
                    target_train,
                    target_test,
                    tl_searchspace,
                    objective,
                    fraction_source,
                )

                # Add metadata for transfer learning models
                for tl_result in tl_results:
                    tl_result.update(
                        {
                            "mc_iter": mc_iter,
                            "n_train_pts": n_train_pts,
                            "fraction_source": fraction_source,
                            "n_source_pts": len(source_subset),
                            "n_test_pts": len(target_test),
                            "source_data_seed": settings.random_seed + mc_iter,
                        }
                    )
                    results.append(tl_result)

                pbar.update(1)

    pbar.close()

    results_df = pd.DataFrame(results)

    return results_df


def _sample_source_data(
    source_data: pd.DataFrame,
    source_tasks: list[str],
    fraction_source: float,
    task_column: str,
    source_data_seed: int,
    stratified_sampling: bool = True,
) -> pd.DataFrame:
    """Sample source data with configurable sampling strategy.

    Args:
        source_data: DataFrame containing all source task data.
        source_tasks: List of source task identifiers.
        fraction_source: Fraction of data to sample.
        task_column: Name of column containing task identifiers.
        source_data_seed: Random seed for reproducible sampling.
        stratified_sampling: If True, samples the same fraction from each source
            task independently (ensures balanced representation across tasks).
            If False, samples the fraction from all source data combined.

    Returns:
        Combined DataFrame with sampled data from source tasks.
    """
    if stratified_sampling:
        # Sample same fraction from each source task
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
    else:
        # Sample fraction from all source data combined
        return source_data.sample(
            frac=fraction_source,
            random_state=source_data_seed,
        )


def _evaluate_model(
    model: GaussianProcessSurrogate,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    searchspace: SearchSpace,
    objective: SingleTargetObjective,
    scenario_name: str,
) -> dict[str, Any]:
    """Train a single model and evaluate its performance.

    Args:
        model: The surrogate model to train.
        train_data: Training data.
        test_data: Test data for evaluation.
        searchspace: Search space for the model.
        objective: Optimization objective.
        scenario_name: Name of the scenario for results.

    Returns:
        Dictionary with scenario name and evaluation metrics.
    """
    target_column = objective._target.name
    train_data_prepared = train_data.copy()
    test_data_prepared = test_data.copy()

    model.fit(
        searchspace=searchspace, objective=objective, measurements=train_data_prepared
    )

    # Evaluate model
    predictions = model.posterior_stats(test_data_prepared, stats=["mean"])
    pred_values = predictions[f"{target_column}_mean"].values
    metrics = _calculate_metrics(
        true_values=np.asarray(test_data[target_column].values),
        pred_values=pred_values,
    )

    result: dict[str, str | float] = {"scenario": scenario_name}
    result.update(metrics)
    return result


def _evaluate_naive_models(
    target_train: pd.DataFrame,
    target_test: pd.DataFrame,
    vanilla_searchspace: SearchSpace,
    tl_searchspace: SearchSpace,
    objective: SingleTargetObjective,
) -> list[dict[str, Any]]:
    """Evaluate both naive model baselines that do not use source data.

    Args:
        target_train: Target task training data.
        target_test: Target task test data.
        vanilla_searchspace: Search space without task parameter.
        tl_searchspace: Search space with task parameter.
        objective: Optimization objective.

    Returns:
        List of evaluation results for naive baselines.
    """
    # Collect evaluation results for models without source data
    results: list[dict[str, Any]] = []

    # Naive GP on reduced search space (no source data, no task parameter)
    results.append(
        _evaluate_model(
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
        _evaluate_model(
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
) -> list[dict[str, Any]]:
    """Evaluate all transfer learning models using source and target data.

    Args:
        source_data: Source task data.
        target_train: Target task training data.
        target_test: Target task test data.
        tl_searchspace: Search space with task parameter.
        objective: Optimization objective.
        fraction_source: Fraction of source data used.

    Returns:
        List of evaluation results for transfer learning models.
    """
    # Collect evaluation results for transfer learning models
    results: list[dict[str, Any]] = []

    combined_data = pd.concat([source_data, target_train])

    for model_suffix, model_class in TL_MODELS.items():
        scenario_name = f"{int(100 * fraction_source)}_{model_suffix}"
        model = model_class()
        results.append(
            _evaluate_model(
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
    pred_values: np.ndarray,
) -> dict[str, float]:
    """Calculate regression metrics for model predictions.

    Args:
        true_values: True target values.
        pred_values: Model predictions..

    Returns:
        Dictionary with metric names as keys and metric values as values.
    """
    results = {}

    for metric_func in REGRESSION_METRICS:
        metric_value = metric_func(true_values, pred_values)
        results[metric_func.__name__] = metric_value

    return results
