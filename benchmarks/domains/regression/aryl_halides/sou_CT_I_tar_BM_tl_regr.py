"""TL regression benchmark for aryl halides data."""

from __future__ import annotations

import pandas as pd

from baybe.objectives import SingleTargetObjective
from baybe.parameters import (
    SubstanceParameter,
    TaskParameter,
)
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from benchmarks.definition import (
    TransferLearningRegression,
    TransferLearningRegressionSettings,
)
from benchmarks.domains.regression.base import (
    run_tl_regression_benchmark,
)
from benchmarks.domains.transfer_learning.aryl_halides.base import load_data


def create_searchspaces(
    data: pd.DataFrame,
) -> tuple[SearchSpace, SearchSpace, str, list[str], str]:
    """Create search spaces for vanilla GP and transfer learning models.

    Args:
        data: DataFrame containing the aryl halide data.

    Returns:
        Tuple containing:
        - vanilla_searchspace: SearchSpace for vanilla GP (no task parameter)
        - tl_searchspace: SearchSpace for transfer learning (with task parameter)
        - name_task: Name of the task parameter
        - source_tasks: List of source task values
        - target_task: Target task value
    """
    # Parameters for both search spaces
    substance_params = [
        SubstanceParameter(
            name=substance,
            data=dict(zip(data[substance], data[f"{substance}_smiles"])),
            encoding="MORDRED",
        )
        for substance in ["base", "ligand", "additive"]
    ]

    # Common parameters
    common_params = substance_params

    # Create vanilla GP search space (no task parameter)
    vanilla_searchspace = SearchSpace.from_product(common_params)

    # Create transfer learning search space (with task parameter)
    name_task = "aryl_halide"
    source_tasks = ["1-chloro-4-(trifluoromethyl)benzene", "2-iodopyridine"]
    target_task = "1-bromo-4-methoxybenzene"

    # Active value for the task parameter
    task_param = TaskParameter(
        name=name_task,
        values=source_tasks + [target_task],
        active_values=[target_task],
    )

    tl_params = common_params + [task_param]
    tl_searchspace = SearchSpace.from_product(tl_params)

    return vanilla_searchspace, tl_searchspace, name_task, source_tasks, target_task


def create_target() -> NumericalTarget:
    """Create the target task for the benchmark."""
    return NumericalTarget(name="yield", mode="MAX")


def create_objective() -> SingleTargetObjective:
    """Create the objective for the benchmark."""
    target = create_target()
    return SingleTargetObjective(target)


def aryl_halide_tl_regression(
    settings: TransferLearningRegressionSettings,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Benchmark function for comparing regression performance of GP vs TL models.

    This benchmark uses aryl halide reactions with different substrates:
    - Source tasks: 1-chloro-4-(trifluoromethyl)benzene and 2-iodopyridine
    - Target task: 1-bromo-4-methoxybenzene

    It trains models with varying amounts of source and target data, and evaluates
    their predictive performance on held-out target data.

    Args:
        settings: The benchmark settings.

    Returns:
        Tuple containing:
        - DataFrame with benchmark results
        - List of metric names used
        - List of model names used
    """
    return run_tl_regression_benchmark(
        settings=settings,
        load_data_fn=load_data,
        create_searchspaces_fn=create_searchspaces,
        create_objective_fn=create_objective,
    )


# Define the benchmark settings
benchmark_config = TransferLearningRegressionSettings(
    random_seed=42,
    num_mc_iterations=1,  # 5,  # Reduced for testing, increase for production
    max_train_points=2,  # 10,  # Adjust based on available data
    source_fractions=[
        0.1,
        0.3,
    ],  # , 0.5],  # Different fractions of source data to test
    noise_std=0.0,  # Not used for real data
    metrics=["RMSE", "R2", "MAE", "LPD", "NLPD"],
)

# Create the benchmark
aryl_halide_tl_regression_benchmark = TransferLearningRegression(
    function=aryl_halide_tl_regression,
    settings=benchmark_config,
)

# For debugging/testing
if __name__ == "__main__":
    from benchmarks.domains.regression.visualization import plot_results

    # Run the benchmark directly
    print("Starting Aryl Halide Transfer Learning Regression Benchmark...")
    result_df, metrics, model_names = aryl_halide_tl_regression(benchmark_config)

    # Print summary
    print(f"Benchmark completed with {len(result_df)} result rows")
    print(f"Metrics evaluated: {metrics}")
    print(f"Models compared: {model_names}")
    print("\nSample results:")
    print(result_df.head())

    # Save results to CSV for further analysis
    result_df.to_csv("aryl_halide_tl_regression_results.csv", index=False)

    # Generate plots using the visualization module
    print("\nGenerating plots...")
    plot_results(
        result_df,
        metrics,
        model_names,
        file_name_prefix="aryl_halide_sou_CT_I_tar_BM_tl_regr",
    )
    print("Plots saved.")
