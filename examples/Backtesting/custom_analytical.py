## Optimizing a Custom Black-Box Function

# This example demonstrates how to optimize a custom black-box function:
# * We create a black-box callable and define the corresponding optimization scope,
# * set up optimization strategies,
# * and compare the resulting trajectories.

### Imports

import os

import numpy as np
import seaborn as sns

from baybe import Campaign
from baybe.parameters.numerical import NumericalContinuousParameter
from baybe.recommenders import RandomRecommender
from baybe.searchspace import SearchSpace
from baybe.simulation import label_columns, simulate_scenarios
from baybe.targets import NumericalTarget
from baybe.utils.plotting import create_example_plots

### Settings

# Before we start, let us collect a few general settings for the example:

SMOKE_TEST = "SMOKE_TEST" in os.environ

BATCH_SIZE = 1
N_MC_ITERATIONS = 2 if SMOKE_TEST else 20
N_DOE_ITERATIONS = 2 if SMOKE_TEST else 30
DIMENSION = 1 if SMOKE_TEST else 10
BOUNDS = (-1, 1)

### Defining the Optimization Problem

# Now, we can define the scope of our optimization problem. Our goal is to optimize
# a high-dimensional quadratic function on a bounded input domain. We first define
# the corresponding inputs and output of the function:

parameters = [
    NumericalContinuousParameter(name=f"x_{k}", bounds=BOUNDS) for k in range(DIMENSION)
]
target = NumericalTarget(name="Target", mode="MIN")


# Based on the above, we construct the black-box callable to be optimized.
# Using the {func}`~baybe.simulation.lookup.label_columns` decorator, we can easily map
# the columns of the raw input/output arrays to our parameter and target objects, which
# creates the required dataframe-based lookup for the optimization loop:


@label_columns([p.name for p in parameters], [target.name])
def sum_of_squares(x: np.ndarray, /) -> np.ndarray:
    """Calculate the sum of squares."""
    return (x**2).sum(axis=1, keepdims=True)


# What remains is to construct the search space and objective for the optimization:

searchspace = SearchSpace.from_product(parameters=parameters)
objective = target.to_objective()

### Creating the Campaigns

# We consider two optimization scenarios, each represented by its own campaign:
# * Optimization using the default recommender
# * A baseline using randomly generated recommendations

default_campaign = Campaign(
    searchspace=searchspace,
    objective=objective,
)
random_campaign = Campaign(
    searchspace=searchspace,
    objective=objective,
    recommender=RandomRecommender(),
)

### Running the Optimization Loop

# Next, we simulate both scenarios using the
# {func}`~baybe.simulation.scenarios.simulate_scenarios` utility,
# which automatically executes several Monte Carlo simulations for each campaign:

scenarios = {
    "Default Recommender": default_campaign,
    "Random Recommender": random_campaign,
}
results = simulate_scenarios(
    scenarios,
    sum_of_squares,
    batch_size=BATCH_SIZE,
    n_doe_iterations=N_DOE_ITERATIONS,
    n_mc_iterations=N_MC_ITERATIONS,
)

### Plotting the Results

# Finally, we compare the trajectories of the campaigns:

ax = sns.lineplot(
    data=results,
    marker="o",
    markersize=10,
    x="Num_Experiments",
    y="Target_CumBest",
    hue="Scenario",
)
create_example_plots(ax=ax, base_name="custom_analytical")
