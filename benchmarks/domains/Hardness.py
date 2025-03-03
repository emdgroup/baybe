# Hardness benchmarking, a maximization task on experimental hardness dataset.

from __future__ import annotations

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from pandas import DataFrame

from baybe.campaign import Campaign
from baybe.parameters import NumericalDiscreteParameter, TaskParameter
from baybe.recommenders.pure.nonpredictive.sampling import RandomRecommender
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget, TargetMode
from benchmarks.definition.convergence import (
    ConvergenceBenchmark,
    ConvergenceBenchmarkSettings,
)

# Set up directory and load datasets
home_dir = os.getcwd()
# Materials Project (MP) bulk modulus dataset
df_mp = pd.read_csv(
    os.path.join(home_dir, "benchmarks", "domains", "mp_bulkModulus_goodOverlap.csv"),
    index_col=0,
)
# Experimental (Exp) hardness dataset
df_exp = pd.read_csv(
    os.path.join(home_dir, "benchmarks", "domains", "exp_hardness_goodOverlap.csv"),
    index_col=0,
)
element_cols = df_exp.columns.to_list()[4:]

# Initialize an empty dataframe to store the integrated hardness values
df_exp_integrated_hardness = pd.DataFrame()

# For each unique composition in df_exp, make a cubic spline interpolation of the hardness vs load curve
for composition_i in df_exp["composition"].unique():
    composition_subset = df_exp[df_exp["composition"] == composition_i]
    # Sort the data by load
    composition_subset = composition_subset.sort_values(by="load")
    composition_subset = composition_subset.drop_duplicates(subset="load")
    if len(composition_subset) < 5:  # Continue to the next composition
        continue

    # Perform cubic spline interpolation of the hardness vs load curve
    spline = sp.interpolate.CubicSpline(
        composition_subset["load"], composition_subset["hardness"]
    )
    # Integrate the spline from the minimum load to the maximum load
    integrated_value = spline.integrate(0.5, 5, extrapolate=True)

    # Make a new dataframe with the element_cols from composition_subset
    composition_summary = composition_subset[
        ["strComposition", "composition"] + element_cols
    ]
    composition_summary = composition_summary.drop_duplicates(subset="composition")
    composition_summary["integratedHardness"] = integrated_value

    df_exp_integrated_hardness = pd.concat(
        [df_exp_integrated_hardness, composition_summary]
    )

# ----- Target function (integrated hardness) -----
df_searchspace_target = df_exp_integrated_hardness[element_cols]
df_searchspace_target["Function"] = "targetFunction"

# Make a lookup table for the task function (integrate hardness) - add the 'integratedHardness' column
df_lookup_target = pd.concat(
    [df_searchspace_target, df_exp_integrated_hardness["integratedHardness"]], axis=1
)
df_lookup_target = df_lookup_target.rename(columns={"integratedHardness": "Target"})

# ----- Source function (voigt bulk modulus) -----
df_searchspace_source = df_mp[element_cols]
df_searchspace_source["Function"] = "sourceFunction"

# Make a lookup table for the source function (voigt bulk modulus) - add the 'vrh' column
df_lookup_source = pd.concat([df_searchspace_source, df_mp["vrh"]], axis=1)
df_lookup_source = df_lookup_source.rename(columns={"vrh": "Target"})

# Combine the search space
df_searchspace = pd.concat([df_searchspace_target, df_searchspace_source])

def hardness(settings: ConvergenceBenchmarkSettings) -> DataFrame:
    """Integrated hardness benchmark, compares across random, default, and no task parameter set up

    Inputs:
        B   discrete    {0.8, 0.66666667, 0.92307692 ...}   |B| = 13
        Sc  discrete    {0.,  0.00384615, 0.01923077 ...}   |Sc| = 26
        Cr  discrete    {0.01, 0.06, 0.1 ...}               |Cr| = 20
        Y   discrete    {0., 0.07307692, 0.05769231 ...}    |Y| = 31
        Zr  discrete    {0., 0.07307692, 0.05769231 ...}    |Zr| = 19
        Gd  discrete    {0., 0.03968254, 0.01587302 ...}    |Gd| = 12
        Hf  discrete    {0., 0.008, 0.02 ...}               |Hf| = 13
        Ta  discrete    {0., 0.006, 0.008 ...}              |Ta| = 17
        W   discrete    {0.19, 0.14, 0.1 ...}               |W| = 30
        Re  discrete    {0., 0.2, 0.33333 ...}              |Re| = 15
    Output: discrete
    Objective: maximization
    """
    parameters = []
    parameters_no_task = []

    # For each column in df_searchspace except the last one, create a NumericalDiscreteParameter
    for column in df_searchspace.columns[:-1]:
        discrete_parameter_i = NumericalDiscreteParameter(
            name=column,
            values=np.unique(df_searchspace[column]),
            tolerance=0.0,
        )
        parameters.append(discrete_parameter_i)
        parameters_no_task.append(discrete_parameter_i)

    task_parameter = TaskParameter(
        name="Function",
        values=["targetFunction", "sourceFunction"],
        active_values=["targetFunction"],
    )
    parameters.append(task_parameter)

    searchspace = SearchSpace.from_dataframe(df_searchspace, parameters=parameters)
    searchspace_no_task = SearchSpace.from_dataframe(
        df_searchspace_target[element_cols], parameters=parameters_no_task
    )

    objective = NumericalTarget(name="Target", mode=TargetMode.MAX).to_objective()

    scenarios: dict[str, Campaign] = {
        "Random Recommender": Campaign(
            searchspace=SearchSpace.from_dataframe(
                df_searchspace_target[element_cols], parameters=parameters_no_task
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

    return simulate_scenarios(
        scenarios,
        df_lookup_target,
        batch_size=settings.batch_size,
        n_doe_iterations=settings.n_doe_iterations,
        n_mc_iterations=settings.n_mc_iterations,
        impute_mode="error",
    )


def hardness_transfer_learning(settings: ConvergenceBenchmarkSettings) -> DataFrame:
    """Integrated hardness benchmark, transfer learning with different initialized data sizes

    Inputs:
        B   discrete    {0.8, 0.66666667, 0.92307692 ...}   |B| = 13
        Sc  discrete    {0.,  0.00384615, 0.01923077 ...}   |Sc| = 26
        Cr  discrete    {0.01, 0.06, 0.1 ...}               |Cr| = 20
        Y   discrete    {0., 0.07307692, 0.05769231 ...}    |Y| = 31
        Zr  discrete    {0., 0.07307692, 0.05769231 ...}    |Zr| = 19
        Gd  discrete    {0., 0.03968254, 0.01587302 ...}    |Gd| = 12
        Hf  discrete    {0., 0.008, 0.02 ...}               |Hf| = 13
        Ta  discrete    {0., 0.006, 0.008 ...}              |Ta| = 17
        W   discrete    {0.19, 0.14, 0.1 ...}               |W| = 30
        Re  discrete    {0., 0.2, 0.33333 ...}              |Re| = 15
    Output: discrete
    Objective: maximization
    """
    parameters = []

    # For each column in df_searchspace except the last one, create a NumericalDiscreteParameter
    for column in df_searchspace.columns[:-1]:
        discrete_parameter_i = NumericalDiscreteParameter(
            name=column,
            values=np.unique(df_searchspace[column]),
            tolerance=0.0,
        )
        parameters.append(discrete_parameter_i)

    task_parameter = TaskParameter(
        name="Function",
        values=["targetFunction", "sourceFunction"],
        active_values=["targetFunction"],
    )
    parameters.append(task_parameter)

    objective = NumericalTarget(name="Target", mode=TargetMode.MAX).to_objective()

    searchspace = SearchSpace.from_dataframe(df_searchspace, parameters=parameters)
    campaign = Campaign(searchspace=searchspace, objective=objective)

    ### ----------- Note: need a elegant way to handle different initial data size ----------- ###
    ### ----------- For now, it is only using n=30 as initial data size ----------- ###
    # Create a list of dataframes with n samples from df_lookup_source to use as initial data
    for n in (2, 4, 6, 30):
        initial_data_i = [
            df_lookup_source.sample(n)
        ]

    return simulate_scenarios(
        {f"{n} Initial Data": campaign},
        df_lookup_target,
        initial_data=initial_data_i,
        batch_size=settings.batch_size,
        n_doe_iterations=settings.n_doe_iterations,
        n_mc_iterations=settings.n_mc_iterations,
        impute_mode="error",
    )

benchmark_config = ConvergenceBenchmarkSettings(
    batch_size=1,
    n_doe_iterations=20,
    n_mc_iterations=5,
)

hardness_benchmark = ConvergenceBenchmark(
    function=hardness,
    best_possible_result=None,
    settings=benchmark_config,
    optimal_function_inputs=None,
)

hardness_transfer_learning_benchmark = ConvergenceBenchmark(
    function=hardness_transfer_learning,
    best_possible_result=None,
    settings=benchmark_config,
    optimal_function_inputs=None,
)

if __name__ == "__main__":
    # Describe the benchmark task
    print(
        "Hardness benchmark is a maximization task on experimental hardness dataset. "
    )
    print(
        "The dataset is downselect to 94 composition with more than 5 hardness values. "
    )
    print(
        "The hardness values are integrated using cubic spline interpolation, and the task is to maximize the integrated hardness. \n"
    )
    print(
        "Hardness benchmark compares across random, default, and no task parameter set up. \n"
    )
    print(
        "Hardness transfer learning benchmark compares across different initialized data sizes. "
    )

    # Visualize the Hardness value histogram
    fig, ax = plt.subplots(
        1, 1, figsize=(8, 5), facecolor="w", edgecolor="k", constrained_layout=True
    )

    # Plot a histogram of the hardness values
    ax.hist(df_exp["hardness"], bins=20)
    ax.set_xlabel("Hardness")
    ax.set_ylabel("Frequency")
    ax.set_title("Integrated Hardness Distribution")
    ax.grid()
