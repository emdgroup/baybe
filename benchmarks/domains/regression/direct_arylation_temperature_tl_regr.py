"""TL regression benchmark for direct arylation data with temperature as task."""

from __future__ import annotations

import pandas as pd

from baybe.objectives import SingleTargetObjective
from baybe.parameters import (
    NumericalDiscreteParameter,
    SubstanceParameter,
    TaskParameter,
)
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from benchmarks.data.utils import DATA_PATH
from benchmarks.definition import (
    TransferLearningRegression,
    TransferLearningRegressionSettings,
)
from benchmarks.domains.regression.base import run_tl_regression_benchmark


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


def create_objective() -> SingleTargetObjective:
    """Create the objective for the benchmark."""
    return SingleTargetObjective(NumericalTarget(name="yield", mode="MAX"))


def direct_arylation_temperature_tl_regr(
    settings: TransferLearningRegressionSettings,
) -> pd.DataFrame:
    """Benchmark function for comparing regression performance of GP vs TL models.

    This benchmark uses reactions at different temperatures:
    - Source tasks: 90°C and 120°C
    - Target task: 105°C

    It trains models with varying amounts of source and target data, and evaluates
    their predictive performance on held-out target data.

    Args:
        settings: The benchmark settings.

    Returns:
        DataFrame with benchmark results
    """
    results_df = run_tl_regression_benchmark(
        settings=settings,
        load_data_fn=load_data,
        create_searchspaces_fn=create_searchspaces,
        create_objective_fn=create_objective,
    )
    return results_df


# Define the benchmark settings
benchmark_config = TransferLearningRegressionSettings(
    random_seed=42,
    num_mc_iterations=30,  # 5,
    max_train_points=10,  # 10,
    source_fractions=[0.01, 0.05, 0.1, 0.2],  # 0.5, 0.7, 0.9],
    noise_std=0.0,  # Not used for real data
)

# Create the benchmark
direct_arylation_temperature_tl_regr_benchmark = TransferLearningRegression(
    function=direct_arylation_temperature_tl_regr, settings=benchmark_config
)
