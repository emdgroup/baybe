"""TL regression benchmark for synthetic function."""

import numpy as np
import pandas as pd
import torch

from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalContinuousParameter, TaskParameter
from baybe.searchspace import SearchSpace
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.surrogates.transfergpbo import MHGPGaussianProcessSurrogate
from baybe.targets import NumericalTarget
from benchmarks.definition import (
    TransferLearningRegression,
    TransferLearningRegressionSettings,
)
from benchmarks.domains.regression.evaluation import evaluate_models


# Function to generate synthetic data for source and target tasks
def generate_data(input_dim, n_points=100, noise_std=0.1, seed=None):
    """Generate synthetic data for source and target tasks."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Generate random input points
    X = np.random.rand(n_points, input_dim)

    # Source task function (e.g., a simple quadratic)
    def source_func(x):
        return np.sum(x**2, axis=1) + np.sin(np.sum(x, axis=1))

    # Target task function (related but different)
    def target_func(x):
        return 0.5 * np.sum(x**2, axis=1) + 0.8 * np.sin(np.sum(x, axis=1) + 0.2)

    # Generate outputs with noise
    y_source = source_func(X) + np.random.normal(0, noise_std, n_points)
    y_target = target_func(X) + np.random.normal(0, noise_std, n_points)

    # Create DataFrames
    source_df = pd.DataFrame({f"x{i}": X[:, i] for i in range(input_dim)})
    source_df["y"] = y_source
    source_df["task"] = "source"

    target_df = pd.DataFrame({f"x{i}": X[:, i] for i in range(input_dim)})
    target_df["y"] = y_target
    target_df["task"] = "target"

    return source_df, target_df


# Create search spaces for vanilla GP and transfer learning
def create_searchspaces(input_dim):
    """Create search spaces for vanilla GP and transfer learning models."""
    # Parameters for both search spaces
    params = [
        NumericalContinuousParameter(f"x{i}", bounds=(0, 1)) for i in range(input_dim)
    ]

    # Create vanilla GP search space (no task parameter)
    vanilla_searchspace = SearchSpace.from_product(params)

    # Create transfer learning search space (with task parameter)
    name_task = "task"
    target_task = "target"  # Active value for the task parameter
    source_tasks = ["source"]  # Source tasks
    task_param = TaskParameter(
        name=name_task, values=source_tasks + [target_task], active_values=[target_task]
    )
    tl_params = params + [task_param]
    tl_searchspace = SearchSpace.from_product(tl_params)

    return vanilla_searchspace, tl_searchspace, name_task, source_tasks, target_task


def create_target():
    """Create a target task for the benchmark."""
    return NumericalTarget(name="y", mode="MIN")


def create_objective() -> SingleTargetObjective:
    """Create the objective for the benchmark."""
    target = NumericalTarget(name="y", mode="MIN")
    return SingleTargetObjective(target)


def transfer_learning_regression(
    settings: TransferLearningRegressionSettings,
) -> pd.DataFrame:
    """Benchmark function for regression performance of vanilla GP vs TL models.

    Args:
        settings: The benchmark settings.

    Returns:
        DataFrame containing benchmark results for all test cases.
    """
    # Set random seed for reproducibility
    np.random.seed(settings.random_seed)
    torch.manual_seed(settings.random_seed)

    # Create objective
    objective = create_objective()
    target_column = objective._target.name

    # Create vanilla search space and fit vanilla GP
    vanilla_searchspace, tl_searchspace, name_task, _, target_task = (
        create_searchspaces(settings.input_dim)
    )

    # Main benchmark loop
    results = []

    for mc_iter in range(settings.num_mc_iterations):
        # Generate new source/target data for each MC iteration
        source_data, target_data = generate_data(
            input_dim=settings.input_dim,
            noise_std=settings.noise_std,
            seed=settings.random_seed + mc_iter,
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
            # Sample source data and keep it constant for all models and training points
            source_subset = source_data.sample(
                frac=fraction_source, random_state=settings.random_seed + mc_iter
            )
            # For each number of target training points
            for n_train_pts in range(1, settings.max_train_points + 1):
                # Split target data into train/test
                target_train = target_data.sample(
                    n=n_train_pts, random_state=settings.random_seed + mc_iter
                )
                target_test = target_data.drop(target_train.index)

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

                # Record results
                results.append(
                    {
                        "mc_iter": mc_iter,
                        "fraction_source": fraction_source,
                        "n_train_pts": n_train_pts,
                        **eval_results,
                    }
                )
    # Extract model names for return value
    model_names = ["vanilla"] + [model_dict["name"] for model_dict in tl_models]

    return pd.DataFrame(results), settings.metrics, model_names


# Define the benchmark settings
benchmark_config = TransferLearningRegressionSettings(
    random_seed=42,
    num_mc_iterations=2,  # 10,
    input_dim=2,
    max_train_points=2,  # 10,
    source_fractions=[0.1, 0.3],  # , 0.5, 0.7, 0.9],
    noise_std=0.1,
    metrics=["RMSE", "R2", "MAE", "LPD", "NLPD"],
)

# Create the benchmark
transfer_learning_regression_benchmark = TransferLearningRegression(
    function=transfer_learning_regression, settings=benchmark_config
)


### DEBUG: Local visualization of the benchmark

if __name__ == "__main__":
    from benchmarks.domains.transfer_learning.regression.visualization import (
        plot_results,
    )

    # Run the benchmark directly
    # Run the benchmark directly
    result, metrics, model_names = transfer_learning_regression(benchmark_config)
    print(f"Benchmark completed with {len(result)} result rows")
    print(result.head())

    # Save results to CSV for further analysis
    result.to_csv("transfer_learning_regression_results.csv", index=False)

    # Visualize results
    plot_results(result, metrics, model_names)
