# Crabnet hyperparameter benchmarking, a minimization task on Crabnet hyperparameters.

# Crabnet hyperparameter function with 20 continuous (treat as discrete for simplicity) and 3 categorical input.
# This code interacts with an external API hosted on Hugging Face Spaces: https://huggingface.co/spaces/AccelerationConsortium/crabnet-hyperparameter

# The external space might be asleep, and this code includes logic to wake it up and retry until it becomes available or a retry limit is reached.

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

# Wake up the Hugging Face space
from gradio_client import Client
from pandas import DataFrame

from baybe.campaign import Campaign
from baybe.parameters import (
    CategoricalParameter,
    NumericalDiscreteParameter,
    TaskParameter,
)
from baybe.recommenders.pure.nonpredictive.sampling import RandomRecommender
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget
from benchmarks.definition.convergence import (
    ConvergenceBenchmark,
    ConvergenceBenchmarkSettings,
)

client = Client("AccelerationConsortium/crabnet-hyperparameter")


def wake_up_hfspace(client, max_retries=2, wait_time=150):
    """Ensure the external Hugging Face space is awake before making predictions.

    Args:
        client: The Gradio Client instance.
        max_retries: Maximum number of retries to wake up the space.
        wait_time: Seconds to wait between retries.

    Raises:
        RuntimeError: If the space does not wake up after max_retries.
    """
    for attempt in range(max_retries):
        try:
            # Attempt a simple request to check if the space is awake
            client.predict(
                0,
                0,
                0,
                0,
                0,
                0.3,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.1,
                0.5,
                "c1_0",
                "c2_0",
                "c3_0",
                0.5,
                api_name="/predict",
            )
            print("Hugging Face space is awake.")
            return
        except Exception:
            print(
                f"Attempt {attempt + 1}: Space is asleep. Retrying in {wait_time} seconds..."
            )
            time.sleep(wait_time)
    raise RuntimeError("Hugging Face space is still asleep after maximum retries.")


wake_up_hfspace(client)


def adv_opt(
    c1,
    c2,
    c3,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15,
    x16,
    x17,
    x18,
    x19,
    x20,
):
    """Optimization function with 20 continuous variables and 3 categorical parameters."""
    result = client.predict(
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        x7,
        x8,
        x9,
        x10,
        x11,
        x12,
        x13,
        x14,
        x15,
        x16,
        x17,
        x18,
        x19,
        x20,  # Continuous variables
        c1,
        c2,
        c3,  # Categorical variables
        0.5,  # Fidelity
        api_name="/predict",
    )
    return result["data"][0][0]  # return y1 value only


# %%
# Run this cell and above if "CrabNet_lookup_testing_y1.csv" and "CrabNet_lookup_training_y2.csv" are not provided
# or if you want to generate data for the benchmark

# def generate_parameters():
#     while True:
#         # Random float values for x1 to x20 between 0.0 and 1.0
#         params = {f"x{i}": np.random.uniform(0.0, 1.0) for i in range(1, 21)}

#         # Random categorical values for c1, c2, c3
#         params["c1"] = np.random.choice(["c1_0", "c1_1"])
#         params["c2"] = np.random.choice(["c2_0", "c2_1"])
#         params["c3"] = np.random.choice(["c3_0", "c3_1", "c3_2"])

#         # Check constraints
#         if params["x19"] < params["x20"] and params["x6"] + params["x15"] <= 1.0:
#             return params

# # Create DataFrame for 20 input data size in number_init_points
# data = [generate_parameters() for _ in range(20)]
# initial_points = pd.DataFrame(data)
# # make sure c1, c2, c3 are str type
# initial_points['c1'] = initial_points['c1'].apply(str)
# initial_points['c2'] = initial_points['c2'].apply(str)
# initial_points['c3'] = initial_points['c3'].apply(str)

# # create a dataframe, that has initial points for y1 and y2
# # add a Target column for y1/y2 value, and a Function column for the fucntion used
# lookup_training_y2 = initial_points.copy()
# lookup_training_y2['Target'] = lookup_training_y2.apply(lambda x: adv_opt_y2(**x), axis=1)
# lookup_training_y2['Function'] = "TrainingY2"

# lookup_testing_y1 = initial_points.copy()
# lookup_testing_y1['Target'] = lookup_testing_y1.apply(lambda x: adv_opt_y1(**x), axis=1)
# lookup_testing_y1['Function'] = "TestingY1"

# # save lookup_training_y2 and lookup_testing_y1 to csv
# lookup_testing_y1.to_csv("CrabNet_lookup_testing_y1.csv", index=False)
# lookup_training_y2.to_csv("CrabNet_lookup_training_y2.csv", index=False)
# %%
# import data
home_dir = os.getcwd()
df_training_y2 = pd.read_csv(
    os.path.join(home_dir, "benchmarks", "domains", "CrabNet_lookup_training_y2.csv")
)
df_testing_y1 = pd.read_csv(
    os.path.join(home_dir, "benchmarks", "domains", "CrabNet_lookup_testing_y1.csv")
)

# concatenate the two dataframes
df_searchspace = pd.concat([df_testing_y1, df_training_y2], ignore_index=True)
# drop the "Target" column
df_searchspace = df_searchspace.drop(columns=["Target"])


def advopt(settings: ConvergenceBenchmarkSettings) -> DataFrame:
    """Crabnet hyperparameter optimization with 20 discrete and 3 categorical input.
    Compare across random, default, and no task parameter settings.

    Inputs:
        x1-x20  discrete        0 ≤ xi ≤ 1 for i ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
        note: x1-x20 are continuous hyperparameters, treat as discrete for simplicity

        c1      categorical     c1 ∈ {c1_0, c1_1}
        c2      categorical     c2 ∈ {c2_0, c2_1}
        c3      categorical     c3 ∈ {c3_0, c3_1, c3_2}
    Output: continuous
    Objective: minimization
    """
    # Define and create the search space
    parameters = []
    parameters_no_task = []
    for col in df_searchspace.columns[:-4]:
        # create a NumericalDiscreteParameter
        discrete_parameter = NumericalDiscreteParameter(
            name=col,
            values=np.unique(df_searchspace[col]),
            tolerance=0.0,
        )
        parameters.append(discrete_parameter)
        parameters_no_task.append(discrete_parameter)

    categorical_parameters = [
        CategoricalParameter(name="c1", values=["c1_0", "c1_1"], encoding="OHE"),
        CategoricalParameter(name="c2", values=["c2_0", "c2_1"], encoding="OHE"),
        CategoricalParameter(
            name="c3", values=["c3_0", "c3_1", "c3_2"], encoding="OHE"
        ),
    ]
    parameters.extend(categorical_parameters)
    parameters_no_task.extend(categorical_parameters)

    task_parameter = TaskParameter(
        name="Function",
        values=["TrainingY2", "TestingY1"],
        active_values=["TestingY1"],
    )

    parameters.append(task_parameter)

    searchspace = SearchSpace.from_dataframe(df_searchspace, parameters=parameters)
    searchspace_no_task = SearchSpace.from_dataframe(
        df_testing_y1.drop(columns=["Target", "Function"]),
        parameters=parameters_no_task,
    )

    # define objective
    objective = NumericalTarget(name="Target", mode="MIN").to_objective()

    # Define the scenarios
    scenarios: dict[str, Campaign] = {
        "Random Recommender": Campaign(
            searchspace=SearchSpace.from_dataframe(
                df_testing_y1.drop(columns=["Target", "Function"]),
                parameters=parameters_no_task,
            ),
            recommender=RandomRecommender(),
            objective=objective,
        ),
        "Default Recommender": Campaign(
            searchspace=searchspace,
            objective=objective,
        ),
        "No Task Parameter": Campaign(
            searchspace=searchspace_no_task,
            objective=objective,
        ),
    }

    # Simulate the scenarios
    return simulate_scenarios(
        scenarios,
        df_testing_y1,
        batch_size=settings.batch_size,
        n_doe_iterations=settings.n_doe_iterations,
        n_mc_iterations=settings.n_mc_iterations,
        impute_mode="error",
    )


def advopt_transfer_learning(settings: ConvergenceBenchmarkSettings) -> DataFrame:
    """Crabnet hyperparameter optimization with 20 discrete and 3 categorical input.
    Transfer learning with different initial data sizes.

    Inputs:
        x1-x20  discrete        0 ≤ xi ≤ 1 for i ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
        note: x1-x20 are continuous hyperparameters, treat as discrete for simplicity

        c1      categorical     c1 ∈ {c1_0, c1_1}
        c2      categorical     c2 ∈ {c2_0, c2_1}
        c3      categorical     c3 ∈ {c3_0, c3_1, c3_2}
    Output: continuous
    Objective: minimization
    """
    # Define and create the search space
    parameters = []
    for col in df_searchspace.columns[:-4]:
        # create a NumericalDiscreteParameter
        discrete_parameter = NumericalDiscreteParameter(
            name=col,
            values=np.unique(df_searchspace[col]),
            tolerance=0.0,
        )
        parameters.append(discrete_parameter)

    categorical_parameters = [
        CategoricalParameter(name="c1", values=["c1_0", "c1_1"], encoding="OHE"),
        CategoricalParameter(name="c2", values=["c2_0", "c2_1"], encoding="OHE"),
        CategoricalParameter(
            name="c3", values=["c3_0", "c3_1", "c3_2"], encoding="OHE"
        ),
    ]
    parameters.extend(categorical_parameters)

    task_parameter = TaskParameter(
        name="Function",
        values=["TrainingY2", "TestingY1"],
        active_values=["TestingY1"],
    )
    parameters.append(task_parameter)

    objective = NumericalTarget(name="Target", mode="MIN").to_objective()

    searchspace = SearchSpace.from_dataframe(df_searchspace, parameters=parameters)

    # Seperate campaign for different initial data size, preventing unintended data overwriting
    scenarios: dict[str, Campaign] = {
        f"{n} Initial Data": Campaign(searchspace=searchspace, objective=objective)
        for n in (50, 100, 500, 700, 1000)
    }

    # Create an iterable of datasets with different initial sizes
    initial_data_sets = [df_training_y2.sample(n) for n in (50, 100, 500, 700, 1000)]

    return simulate_scenarios(
        scenarios,
        df_testing_y1,
        initial_data=initial_data_sets,
        batch_size=settings.batch_size,
        n_doe_iterations=settings.n_doe_iterations,
        impute_mode="error",
    )


benchmark_config = ConvergenceBenchmarkSettings(
    batch_size=1,
    n_doe_iterations=30,
    n_mc_iterations=5,
)

# Define the benchmark
crabnet_benchmark = ConvergenceBenchmark(
    function=advopt,
    settings=benchmark_config,
    optimal_target_values=None,
)

crabnet_transfer_learning_benchmark = ConvergenceBenchmark(
    function=advopt_transfer_learning,
    settings=benchmark_config,
    optimal_target_values=None,
)


if __name__ == "__main__":
    #  Describe the benchmark task
    print(
        "CrabNet optimization is a minimization task by tuning 20 numerical and 3 categorical hyperparameters."
    )
    print(
        "The numerical hyperparameters include number of attention layers, learning rate, step size of epochs, etc."
    )
    print(
        "Details can be found in Table 1 of Baird, S. G.; Liu, M.; Sparks, T. D. High-Dimensional Bayesian Optimization of 23 Hyperparameters over 100 Iterations for an Attention-Based Network to Predict Materials Property: A Case Study on CrabNet Using Ax Platform and SAASBO. Computational Materials Science 2022, 211, 111505."
    )
    print(
        "The categorical hyperparameters include boolean values for bias residual network, loss function, and elemental feature vector."
    )
    print(
        "The numerical hyperparameters are normalized to range [0.0, 1.0], while the categorical hyperparameters are one-hot encoded. \n"
    )
    print(
        "The objective is to minimize y1, RMSE, of the CrabNet hyperparameter function. If y1 is greater than 0.2, the result is coonsider bad. \n"
    )
    print(
        "CrabNet benchmark compares across random, default, and no task parameter set up. \n"
    )
    print(
        "CrabNet transfer learning benchmark compares across different initialized data sizes. "
    )
