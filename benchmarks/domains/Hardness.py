"""Synthetic function with two continuous and one discrete input."""

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

# dfMP = pd.read_csv(
#     os.path.join(strHomeDir, "data", "processed", "mp_bulkModulus_goodOverlap.csv"), index_col=0
# )

dfExp = pd.read_csv(
    os.path.join(strHomeDir, "benchmarks", "domains", "exp_hardness.csv"), index_col=0
)

lstElementCols = dfExp.columns.to_list()[4:]

# ----- FUTHER CLEAN THE DATA BASED ON THE EDA -----

# initialize an empty dataframe to store the integrated hardness values
dfExp_integratedHardness = pd.DataFrame()

# for each unique composition in dfExp, make a cubic spline interpolation of the hardness vs load curve
for strComposition_temp in dfExp["composition"].unique():
    # get the data for the composition
    dfComposition_temp = dfExp[dfExp["composition"] == strComposition_temp]
    # sort the data by load
    dfComposition_temp = dfComposition_temp.sort_values(by="load")
    # if there are any duplicate values for load, drop them
    dfComposition_temp = dfComposition_temp.drop_duplicates(subset="load")
    # if there are less than 5 values, continue to the next composition
    if len(dfComposition_temp) < 5:
        continue

    # make a cubic spline interpolation of the hardness vs load curve
    spSpline_temp = sp.interpolate.CubicSpline(dfComposition_temp["load"], dfComposition_temp["hardness"])
    # integrate the spline from the minimum load to the maximum load
    fltIntegral_temp = spSpline_temp.integrate(0.5, 5, extrapolate = True)

    # make a new dataframe with the lstElementCols from dfComposition_temp
    dfComposition_temp = dfComposition_temp[['strComposition', 'composition'] + lstElementCols]
    dfComposition_temp = dfComposition_temp.drop_duplicates(subset='composition')
    # add a column to dfComposition_temp called 'integratedHardness' and set all values to fltIntegral_temp
    dfComposition_temp["integratedHardness"] = fltIntegral_temp

    # append dfComposition_temp to dfExp_integratedHardness
    dfExp_integratedHardness = pd.concat([dfExp_integratedHardness, dfComposition_temp])


# ----- CREATE _lookup FOR THE SEARCH SPACE -----
# ----- TARGET FUNCTION (INTEGRATED HARDNESS) -----
# make a dataframe for the task function (integrated hardness)
dfSearchSpace_target = dfExp_integratedHardness[lstElementCols]
# add a column to dfSearchSpace_task called 'Function' and set all values to 'taskFunction'
dfSearchSpace_target["Function"] = "targetFunction"

# make a lookup table for the task function (integrate hardness) - add the 'integratedHardness' column from dfExp to dfSearchSpace_task
dfLookupTable_target = pd.concat([dfSearchSpace_target, dfExp_integratedHardness["integratedHardness"]], axis=1)
# make the 'integrate hardness' column the 'Target' column
_lookup = dfLookupTable_target.rename(columns={"integratedHardness":"Target"})

dfSearchSpace = dfSearchSpace_target


def hardness(settings: ConvergenceExperimentSettings) -> DataFrame:
    """Exp hardness dataset.

    Inputs: composition

    Output: discrete
    Objective: Maximization
    Optimal Inputs:
        ----
        ----
    Optimal Output:
        ----
    """

    lstParameters_bb = []
    # for each column in dfSearchSpace except the last one, create a NumericalDiscreteParameter
    for strCol_temp in dfSearchSpace.columns[:-1]:
        # create a NumericalDiscreteParameter
        bbParameter_temp = NumericalDiscreteParameter(
            name=strCol_temp,
            values=np.unique(dfSearchSpace[strCol_temp]),
            tolerance=0.0,
        )
        # append the parameter to the list of parameters
        lstParameters_bb.append(bbParameter_temp)

    search_space = SearchSpace.from_dataframe(dfSearchSpace, parameters=lstParameters_bb)
    objective = NumericalTarget(name="target", mode=TargetMode.MAX).to_objective()


    scenarios: dict[str, Campaign] = {
        "Random Recommender": Campaign(
            searchspace=search_space,
            recommender=RandomRecommender(),
            objective=objective,
        ),
        "Default Recommender": Campaign(
            searchspace=search_space,
            objective=objective,
        ),
    }

    return simulate_scenarios(
        scenarios,
        _lookup,
        batch_size=settings.batch_size,
        n_doe_iterations=settings.n_doe_iterations,
        n_mc_iterations=settings.n_mc_iterations,
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


if __name__ == "__main__":

    # describe the benchmark task 
    print("Hardness benchmark is a maximization task on experimental hardness dataset. ")
    print("The dataset is downselect to 94 composition with more than 5 hardness values. ")
    print("The hardness values are integrated using cubic spline interpolation, and the task is to maximize the integrated hardness. ")


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
    ax.set_title("Hardness Distribution")

    # add a grid
    ax.grid()