## Simulation loop using a BoTorch test function

# This example shows a simulation loop for a single target with a BoTorch test function as lookup.

# This example assumes some basic familiarity with using BayBE and how to use BoTorch test
# functions in discrete searchspaces.
# We thus refer to
# 1. [`campaign`](./../Basics/campaign.md) for a basic example on how to use BayBE and
# 2. [`discrete_space`](./../Searchspaces/discrete_space.md) for details on using a
# BoTorch test function.

### Imports

import os

import numpy as np
import seaborn as sns
from botorch.test_functions import Rastrigin

from baybe import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalDiscreteParameter
from baybe.recommenders import RandomRecommender
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget
from baybe.utils.botorch_wrapper import botorch_function_wrapper
from baybe.utils.plotting import create_example_plots

### Parameters for a full simulation loop

# For the full simulation, we need to define the number of Monte Carlo runs
# and the number of experiments to be conducted per run.

SMOKE_TEST = "SMOKE_TEST" in os.environ

N_MC_ITERATIONS = 2 if SMOKE_TEST else 30
N_DOE_ITERATIONS = 2 if SMOKE_TEST else 15
BATCH_SIZE = 1 if SMOKE_TEST else 3
POINTS_PER_DIM = 10

### Defining the test function

# See [`discrete_space`](./../Searchspaces/discrete_space.md) for details.

DIMENSION = 4
TestFunctionClass = Rastrigin

if not hasattr(TestFunctionClass, "dim"):
    TestFunction = TestFunctionClass(dim=DIMENSION)
else:
    print(
        f"\nYou choose a dimension of {DIMENSION} for the test function"
        f"{TestFunctionClass}. However, this function can only be used in "
        f"{TestFunctionClass().dim} dimension, so the provided dimension is replaced."
    )
    TestFunction = TestFunctionClass()
    DIMENSION = TestFunctionClass().dim

BOUNDS = TestFunction.bounds
WRAPPED_FUNCTION = botorch_function_wrapper(test_function=TestFunction)

### Creating the searchspace and the objective

parameters = [
    NumericalDiscreteParameter(
        name=f"x_{k+1}",
        values=list(
            np.linspace(
                BOUNDS[0, k],
                BOUNDS[1, k],
                POINTS_PER_DIM,
            )
        ),
        tolerance=0.01,
    )
    for k in range(DIMENSION)
]

searchspace = SearchSpace.from_product(parameters=parameters)
objective = SingleTargetObjective(target=NumericalTarget(name="Target", mode="MIN"))

### Constructing campaigns

seq_greedy_EI_campaign = Campaign(
    searchspace=searchspace,
    objective=objective,
)
random_campaign = Campaign(
    searchspace=searchspace,
    recommender=RandomRecommender(),
    objective=objective,
)

### Performing the simulation loop

# We use [simulate_scenarios](baybe.simulation.scenarios.simulate_scenarios) to simulate a full experiment.

scenarios = {
    "Sequential greedy EI": seq_greedy_EI_campaign,
    "Random": random_campaign,
}
results = simulate_scenarios(
    scenarios,
    WRAPPED_FUNCTION,
    batch_size=BATCH_SIZE,
    n_doe_iterations=N_DOE_ITERATIONS,
    n_mc_iterations=N_MC_ITERATIONS,
)

# We use the plotting utility to create plots.

ax = sns.lineplot(
    data=results,
    marker="o",
    markersize=10,
    x="Num_Experiments",
    y="Target_CumBest",
    hue="Scenario",
)
create_example_plots(ax=ax, base_name="botorch_analytical")
