# Hardness benchmarking, a maximization task on experimental hardness dataset. 

from __future__ import annotations

from typing import TYPE_CHECKING

from pandas import DataFrame
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns

from baybe.campaign import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalDiscreteParameter, TaskParameter
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget
from baybe.utils.random import set_random_seed
from baybe.recommenders.pure.nonpredictive.sampling import RandomRecommender
from baybe.targets import NumericalTarget, TargetMode
from benchmarks.definition import (
    Benchmark,
    ConvergenceExperimentSettings,
)

# Set up directory and load datasets
HomeDir = os.getcwd()
# Materials Project (MP) bulk modulus dataset
dfMP = pd.read_csv(os.path.join(HomeDir, "benchmarks", "domains", "mp_bulkModulus_goodOverlap.csv"), index_col=0)
# Experimental (Exp) hardness dataset
dfExp = pd.read_csv(os.path.join(HomeDir, "benchmarks", "domains", "exp_hardness_goodOverlap.csv"), index_col=0)
elementCols = dfExp.columns.to_list()[4:]

# Initialize an empty dataframe to store the integrated hardness values
dfExp_integratedHardness = pd.DataFrame()

# For each unique composition in dfExp, make a cubic spline interpolation of the hardness vs load curve
for composition_i in dfExp["composition"].unique():
    composition_subset = dfExp[dfExp["composition"] == composition_i]
    # Sort the data by load
    composition_subset = composition_subset.sort_values(by="load")
    composition_subset = composition_subset.drop_duplicates(subset="load")
    if len(composition_subset) < 5:     # Continue to the next composition
        continue

    # Perform cubic spline interpolation of the hardness vs load curve
    spline = sp.interpolate.CubicSpline(composition_subset["load"], composition_subset["hardness"])
    # Integrate the spline from the minimum load to the maximum load
    integrated_value = spline.integrate(0.5, 5, extrapolate = True)

    # Make a new dataframe with the elementCols from composition_subset
    composition_summary = composition_subset[['strComposition', 'composition'] + elementCols]
    composition_summary = composition_summary.drop_duplicates(subset='composition')
    composition_summary["integratedHardness"] = integrated_value

    dfExp_integratedHardness = pd.concat([dfExp_integratedHardness, composition_summary])

# ----- Target function (integrated hardness) -----
dfSearchSpace_target = dfExp_integratedHardness[elementCols]
dfSearchSpace_target["Function"] = "targetFunction"

# Make a lookup table for the task function (integrate hardness) - add the 'integratedHardness' column from dfExp to dfSearchSpace_task
dfLookupTable_target = pd.concat([dfSearchSpace_target, dfExp_integratedHardness["integratedHardness"]], axis=1)
dfLookupTable_target = dfLookupTable_target.rename(columns={"integratedHardness":"Target"})

# ----- Source function (voigt bulk modulus) -----
dfSearchSpace_source = dfMP[elementCols]
dfSearchSpace_source["Function"] = "sourceFunction"

# Make a lookup table for the source function (voigt bulk modulus) - add the 'vrh' column from dfMP to dfSearchSpace_source
dfLookupTable_source = pd.concat([dfSearchSpace_source, dfMP["vrh"]], axis=1)
dfLookupTable_source = dfLookupTable_source.rename(columns={"vrh": "Target"})

# Combine the search space
dfSearchSpace = pd.concat([dfSearchSpace_target, dfSearchSpace_source])

def hardness(settings: ConvergenceExperimentSettings) -> DataFrame:
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
    Objective: Maximization
    """

    parameters = []
    parameters_noTask = []

    # For each column in dfSearchSpace except the last one, create a NumericalDiscreteParameter
    for column in dfSearchSpace.columns[:-1]:
        parameter_i = NumericalDiscreteParameter(
            name=column,
            values=np.unique(dfSearchSpace[column]),
            tolerance=0.0,
        )
        parameters.append(parameter_i)
        parameters_noTask.append(parameter_i)
    
    # Create TaskParameter
    taskParameter = TaskParameter(
        name="Function",
        values=["targetFunction", "sourceFunction"],
        active_values=["targetFunction"],
    )   
    parameters.append(taskParameter)

    search_space = SearchSpace.from_dataframe(dfSearchSpace, parameters=parameters)
    SearchSpace_noTask = SearchSpace.from_dataframe(dfSearchSpace_target[elementCols], parameters=parameters_noTask)
    
    objective = NumericalTarget(name="Target", mode=TargetMode.MAX).to_objective()

    scenarios: dict[str, Campaign] = {
        "Random Recommender": Campaign(
            searchspace=SearchSpace.from_dataframe(
                dfSearchSpace_target[elementCols],
                parameters=parameters_noTask
            ),
            recommender=RandomRecommender(),
            objective=objective,
        ),
        "Default Recommender": Campaign(
            searchspace=search_space,
            objective=objective,
        ),
        "No TaskParameter": Campaign(
            searchspace=SearchSpace_noTask,
            objective=objective,
        ),
    }

    return simulate_scenarios(
        scenarios,
        dfLookupTable_target,
        batch_size=settings.batch_size,
        n_doe_iterations=settings.n_doe_iterations,
        n_mc_iterations=settings.n_mc_iterations,
        impute_mode="error",
    )


def hardness_transfer_learning(settings: ConvergenceExperimentSettings) -> DataFrame:
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
    Objective: Maximization
    """

    parameters = []
    parameters_noTask = []

    # For each column in dfSearchSpace except the last one, create a NumericalDiscreteParameter
    for column in dfSearchSpace.columns[:-1]:
        parameter_i = NumericalDiscreteParameter(
            name=column,
            values=np.unique(dfSearchSpace[column]),
            tolerance=0.0,
        )
        parameters.append(parameter_i)
        parameters_noTask.append(parameter_i)
    
    # Create TaskParameter
    taskParameter = TaskParameter(
        name="Function",
        values=["targetFunction", "sourceFunction"],
        active_values=["targetFunction"],
    )   
    parameters.append(taskParameter)

    objective = NumericalTarget(name="Target", mode=TargetMode.MAX).to_objective()

    search_space = SearchSpace.from_dataframe(dfSearchSpace, parameters=parameters)
    campaign = Campaign(searchspace=search_space, objective=objective)

    # Use diff init data size ----------------------
    # Create a list of dataframes with n samples from dfLookupTable_source to use as initial data
    for n in (2, 4, 6, 30):
        initialData_i = [dfLookupTable_source.sample(n) for _ in range(settings.n_mc_iterations)]

    return simulate_scenarios(
        {f"{n} Initial Data": campaign},
        dfLookupTable_target,
        initial_data=initialData_i, 
        batch_size=settings.batch_size,
        n_doe_iterations=settings.n_doe_iterations,
        impute_mode="error",
    )

benchmark_config = ConvergenceExperimentSettings(
    batch_size=1,
    n_doe_iterations=20,
    n_mc_iterations=5,
)

hardness_benchmark = Benchmark(
    function=hardness,
    best_possible_result=None,
    settings=benchmark_config,
    optimal_function_inputs=None,
)

hardness_transfer_learning_benchmark = Benchmark(
    function=hardness_transfer_learning,
    best_possible_result=None,
    settings=benchmark_config,
    optimal_function_inputs=None,
)


if __name__ == "__main__":

    # Describe the benchmark task 
    print("Hardness benchmark is a maximization task on experimental hardness dataset. ")
    print("The dataset is downselect to 94 composition with more than 5 hardness values. ")
    print("The hardness values are integrated using cubic spline interpolation, and the task is to maximize the integrated hardness. \n")
    print("Hardness benchmark compares across random, default, and no task parameter set up. \n")
    print("Hardness transfer learning benchmark compares across different initialized data sizes. ")

    # Visualize the Hardness value histogram
    fig, ax = plt.subplots(
        1, 1,
        figsize=(8, 5),
        facecolor='w',
        edgecolor='k',
        constrained_layout = True
    )

    # Plot a histogram of the hardness values
    ax.hist(dfExp["hardness"], bins=20)
    ax.set_xlabel("Hardness")
    ax.set_ylabel("Frequency")
    ax.set_title("Integrated Hardness Distribution")
    ax.grid()