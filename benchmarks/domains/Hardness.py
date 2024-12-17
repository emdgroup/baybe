"""
@Time    :   2024/09/30 11:17:24
@Author  :   Daniel Persaud
@Version :   1.0
@Contact :   da.persaud@mail.utoronto.ca
@Desc    :   Hardness benchmarking, a maximization task on experimental hardness dataset. 
"""

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


# IMPORT AND PREPROCESS DATA------------------------------------------------------------------------------
strHomeDir = os.getcwd()
dfMP = pd.read_csv(
    os.path.join(strHomeDir, "benchmarks", "domains", "mp_bulkModulus_goodOverlap.csv"), index_col=0
)
dfExp = pd.read_csv(
    os.path.join(strHomeDir, "benchmarks", "domains", "exp_hardness_goodOverlap.csv"), index_col=0
)
lstElementCols = dfExp.columns.to_list()[4:]

# ----- FUTHER CLEAN THE DATA BASED ON THE EDA -----
# initialize an empty dataframe to store the integrated hardness values
dfExp_integratedHardness = pd.DataFrame()

# for each unique composition in dfExp, make a cubic spline interpolation of the hardness vs load curve
for strComposition_temp in dfExp["composition"].unique():
    dfComposition_temp = dfExp[dfExp["composition"] == strComposition_temp]
    # sort the data by load
    dfComposition_temp = dfComposition_temp.sort_values(by="load")
    dfComposition_temp = dfComposition_temp.drop_duplicates(subset="load")
    if len(dfComposition_temp) < 5:     # continue to the next composition
        continue

    # make a cubic spline interpolation of the hardness vs load curve
    spSpline_temp = sp.interpolate.CubicSpline(dfComposition_temp["load"], dfComposition_temp["hardness"])
    # integrate the spline from the minimum load to the maximum load
    fltIntegral_temp = spSpline_temp.integrate(0.5, 5, extrapolate = True)

    # make a new dataframe with the lstElementCols from dfComposition_temp
    dfComposition_temp = dfComposition_temp[['strComposition', 'composition'] + lstElementCols]
    dfComposition_temp = dfComposition_temp.drop_duplicates(subset='composition')
    dfComposition_temp["integratedHardness"] = fltIntegral_temp

    dfExp_integratedHardness = pd.concat([dfExp_integratedHardness, dfComposition_temp])

# ----- TARGET FUNCTION (INTEGRATED HARDNESS) -----
# make a dataframe for the task function (integrated hardness)
dfSearchSpace_target = dfExp_integratedHardness[lstElementCols]
dfSearchSpace_target["Function"] = "targetFunction"

# make a lookup table for the task function (integrate hardness) - add the 'integratedHardness' column from dfExp to dfSearchSpace_task
dfLookupTable_target = pd.concat([dfSearchSpace_target, dfExp_integratedHardness["integratedHardness"]], axis=1)
dfLookupTable_target = dfLookupTable_target.rename(columns={"integratedHardness":"Target"})

# ----- SOURCE FUNCTION (VOIGT BULK MODULUS) -----
# make a dataframe for the source function (voigt bulk modulus)
dfSearchSpace_source = dfMP[lstElementCols]
dfSearchSpace_source["Function"] = "sourceFunction"

# make a lookup table for the source function (voigt bulk modulus) - add the 'vrh' column from dfMP to dfSearchSpace_source
dfLookupTable_source = pd.concat([dfSearchSpace_source, dfMP["vrh"]], axis=1)
dfLookupTable_source = dfLookupTable_source.rename(columns={"vrh": "Target"})

# concatenate the two dataframes
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

    lstParameters_bb = []
    lstParameters_bb_noTask = []

    # for each column in dfSearchSpace except the last one, create a NumericalDiscreteParameter
    for strCol_temp in dfSearchSpace.columns[:-1]:
        bbParameter_temp = NumericalDiscreteParameter(
            name=strCol_temp,
            values=np.unique(dfSearchSpace[strCol_temp]),
            tolerance=0.0,
        )
        # append the parameter to the list of parameters
        lstParameters_bb.append(bbParameter_temp)
        lstParameters_bb_noTask.append(bbParameter_temp)
    
    # create a TaskParameter
    bbTaskParameter = TaskParameter(
        name="Function",
        values=["targetFunction", "sourceFunction"],
        active_values=["targetFunction"],
    )   

    # append the taskParameter to the list of parameters
    lstParameters_bb.append(bbTaskParameter)

    search_space = SearchSpace.from_dataframe(dfSearchSpace, parameters=lstParameters_bb)
    SearchSpace_noTask = SearchSpace.from_dataframe(dfSearchSpace_target[lstElementCols], parameters=lstParameters_bb_noTask)
    
    objective = NumericalTarget(name="Target", mode=TargetMode.MAX).to_objective()

    scenarios: dict[str, Campaign] = {
        "Random Recommender": Campaign(
            searchspace=SearchSpace.from_dataframe(
                dfSearchSpace_target[lstElementCols],
                parameters=lstParameters_bb_noTask
            ),
            recommender=RandomRecommender(),
            objective=objective,
        ),
        "Default Recommender": Campaign(
            searchspace=SearchSpace.from_dataframe(
                dfSearchSpace, 
                parameters=lstParameters_bb,
            ),
            objective=objective,
        ),
        "noTask_bb": Campaign(
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

    lstParameters_bb = []
    lstParameters_bb_noTask = []

    # for each column in dfSearchSpace except the last one, create a NumericalDiscreteParameter
    for strCol_temp in dfSearchSpace.columns[:-1]:
        bbParameter_temp = NumericalDiscreteParameter(
            name=strCol_temp,
            values=np.unique(dfSearchSpace[strCol_temp]),
            tolerance=0.0,
        )
        # append the parameter to the list of parameters
        lstParameters_bb.append(bbParameter_temp)
        lstParameters_bb_noTask.append(bbParameter_temp)
    
    # create a TaskParameter
    bbTaskParameter = TaskParameter(
        name="Function",
        values=["targetFunction", "sourceFunction"],
        active_values=["targetFunction"],
    )   

    # append the taskParameter to the list of parameters
    lstParameters_bb.append(bbTaskParameter)

    objective = NumericalTarget(name="Target", mode=TargetMode.MAX).to_objective()

    for n in (2, 4, 6, 30):
        bbSearchSpace = SearchSpace.from_dataframe(dfSearchSpace, parameters=lstParameters_bb)
        bbCampaign_temp = Campaign(
            searchspace=bbSearchSpace,
            objective=objective)
        # create a list of dataframes with n samples from dfLookupTable_source to use as initial data
        lstInitialData_temp = [dfLookupTable_source.sample(n) for _ in range(settings.n_mc_iterations)]

    return simulate_scenarios(
        {f"{n} Initial Data": bbCampaign_temp},
        dfLookupTable_target,
        initial_data=lstInitialData_temp, 
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

    # describe the benchmark task 
    print("Hardness benchmark is a maximization task on experimental hardness dataset. ")
    print("The dataset is downselect to 94 composition with more than 5 hardness values. ")
    print("The hardness values are integrated using cubic spline interpolation, and the task is to maximize the integrated hardness. ")
    print("")
    print("Hardness benchmark compares across random, default, and no task parameter set up. ")
    print("")
    print("Hardness transfer learning benchmark compares across different initialized data sizes. ")


    #  Visualize the Hardness value histogram
    # initialize a subplot with 1 row and 1 column
    fig, ax = plt.subplots(
        1, 1,
        figsize=(8, 5),
        facecolor='w',
        edgecolor='k',
        constrained_layout = True
    )

    # plot a histogram of the hardness values
    ax.hist(dfExp["hardness"], bins=20)

    # add a title, x-aixs label, and y-axis label
    ax.set_xlabel("Hardness")
    ax.set_ylabel("Frequency")
    ax.set_title("Integrated Hardness Distribution")

    # add a grid
    ax.grid()