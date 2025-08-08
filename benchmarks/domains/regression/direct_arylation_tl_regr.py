"""TL regression benchmark for direct arylation data with temperature as task."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from baybe.objectives import SingleTargetObjective
from baybe.parameters import (
    NumericalDiscreteParameter,
    SubstanceParameter,
    TaskParameter,
)
from baybe.searchspace import SearchSpace
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.surrogates.transfergpbo import MHGPGaussianProcessSurrogate
from baybe.targets import NumericalTarget
from benchmarks.data.utils import DATA_PATH
from benchmarks.definition import (
    TransferLearningRegression,
    TransferLearningRegressionSettings,
)
from benchmarks.domains.regression.base import (
    evaluate_models,
    run_tl_regression_benchmark,
)


def load_data() -> pd.DataFrame:
    """Load direct arylation data for the benchmark."""
    data = pd.read_table(
        DATA_PATH / "direct_arylation" / "data.csv",
        sep=",",
        index_col=0,
        dtype={"Temp_C": str},  # Ensure temperature is treated as a string
    )

    # Keep only necessary columns
    keep_columns = [
        "Base",
        "Ligand",
        "Solvent",
        "Base_SMILES",
        "Ligand_SMILES",
        "Solvent_SMILES",
        "Concentration",
        "Temp_C",
        "yield",
    ]

    return data[keep_columns]


def create_searchspaces(data: pd.DataFrame) -> tuple[SearchSpace, SearchSpace]:
    """Create search spaces for vanilla GP and transfer learning models."""
    # Parameters for both search spaces
    substance_params = [
        SubstanceParameter(
            name=substance,
            data=dict(zip(data[substance], data[f"{substance}_SMILES"])),
            encoding="RDKIT2DDESCRIPTORS",
        )
        for substance in ["Base", "Ligand", "Solvent"]
    ]

    concentration_param = NumericalDiscreteParameter(
        name="Concentration",
        values=sorted(data["Concentration"].unique()),
    )

    # Common parameters
    common_params = substance_params + [concentration_param]

    # Create vanilla GP search space (no task parameter)
    vanilla_searchspace = SearchSpace.from_product(common_params)

    # Create transfer learning search space (with task parameter)
    name_task = "Temp_C"
    source_tasks = ["90", "120"]
    target_task = "105"  # Active value for the task parameter
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
    target = NumericalTarget(name="yield", mode="MAX")
    return SingleTargetObjective(target)


def direct_arylation_tl_regression(
    settings: TransferLearningRegressionSettings,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Benchmark function for comparing regression performance of GP vs TL models.

    This benchmark uses reactions at different temperatures:
    - Source tasks: 90°C and 120°C
    - Target task: 105°C

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


def direct_arylation_tl_regression_old(
    settings: TransferLearningRegressionSettings,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Benchmark function for comparing regression performance of GP vs TL models.

    This benchmark uses reactions at different temperatures:
    - Source tasks: 90°C and 120°C
    - Target task: 105°C

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
    # Set random seed for reproducibility
    np.random.seed(settings.random_seed)
    torch.manual_seed(settings.random_seed)

    # Create target
    objective = create_objective()
    target_column = objective._target.name

    # Create search spaces
    data = load_data()
    vanilla_searchspace, tl_searchspace, name_task, source_tasks, target_task = (
        create_searchspaces(data)
    )

    # Split data into source and target
    source_data = data[data["Temp_C"].isin(source_tasks)]
    target_data = data[data["Temp_C"] == target_task]

    # Main benchmark loop
    results = []

    for mc_iter in range(settings.num_mc_iterations):
        print(f"Monte Carlo iteration {mc_iter + 1}/{settings.num_mc_iterations}")

        # Create train/test split for target task
        target_indices = np.random.permutation(len(target_data))
        max_train_points = min(
            settings.max_train_points, len(target_data) - 10
        )  # Ensure at least 10 test points

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


# Define the benchmark settings
benchmark_config = TransferLearningRegressionSettings(
    random_seed=42,
    num_mc_iterations=1,  # 10,
    max_train_points=2,  # 15,
    source_fractions=[0.1, 0.3],  # , 0.5, 0.7, 0.9],
    noise_std=0.0,  # Not used for real data
    metrics=["RMSE", "R2", "MAE", "LPD", "NLPD"],
)

# Create the benchmark
direct_arylation_tl_regression_benchmark = TransferLearningRegression(
    function=direct_arylation_tl_regression, settings=benchmark_config
)


if __name__ == "__main__":
    # Import the plotting function from the original module
    from benchmarks.domains.regression.visualization import (
        plot_results,
    )

    # Run the benchmark directly
    print("Starting Direct Arylation Transfer Learning Regression Benchmark...")

    # Run the benchmark
    result_df, metrics, model_names = direct_arylation_tl_regression(benchmark_config)

    # Print summary
    print(f"Benchmark completed with {len(result_df)} result rows")
    print(f"Metrics evaluated: {metrics}")
    print(f"Models compared: {model_names}")
    print("\nSample results:")
    print(result_df.head())

    # Save results to CSV for further analysis
    result_df.to_csv("direct_arylation_tl_regression_results.csv", index=False)
    print("\nResults saved to 'direct_arylation_tl_regression_results.csv'")

    # Generate plots using the imported function
    print("\nGenerating plots...")
    plot_results(
        result_df, metrics, model_names, file_name_prefix="direct_arylation_tl_regr"
    )
    print("Plots saved.")
