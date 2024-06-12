## Example for full simulation loop using a custom analytical test function

# This example shows a simulation loop for a single target with a custom test function as lookup.
# That is, we perform several Monte Carlo runs with several iterations.
# In addition, we also store and display the results.

# This example assumes some basic familiarity with using BayBE and how to use BoTorch test
# functions in discrete searchspaces.
# For further details, we thus refer to
# - [`campaign`](./../Basics/campaign.md) for a basic example on how to use BayBE and
# - [here](./../Searchspaces/continuous_space_custom_function.md) for how to use a custom function.

### Necessary imports for this example

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from baybe import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalDiscreteParameter
from baybe.recommenders import (
    BotorchRecommender,
    RandomRecommender,
    TwoPhaseMetaRecommender,
)
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget

### Parameters for a full simulation loop

# For the full simulation, we need to define some additional parameters.
# These are the number of Monte Carlo runs and the number of experiments to be conducted per run.

# The parameter `POINTS_PER_DIM` controls the number of points per dimension.
# Note that the searchspace will have `POINTS_PER_DIM**DIMENSION` many points.

SMOKE_TEST = "SMOKE_TEST" in os.environ

N_MC_ITERATIONS = 2 if SMOKE_TEST else 5
N_DOE_ITERATIONS = 2 if SMOKE_TEST else 5
DIMENSION = 4
BOUNDS = [(-2, 2), (-2, 2), (-2, 2), (-2, 2)]
POINTS_PER_DIM = 3 if SMOKE_TEST else 10

### Defining the test function

# See [here](./../Searchspaces/continuous_space_custom_function.md) for details.


def sum_of_squares(*x: float) -> float:
    """Calculate the sum of squares."""
    res = 0
    for y in x:
        res += y**2
    return res


### Creating the searchspace and the objective

# As we expect it to be the most common use case, we construct a purely discrete space here.
# Details on how to adjust this for other spaces can be found in the searchspace examples.

parameters = [
    NumericalDiscreteParameter(
        name=f"x_{k+1}",
        values=list(np.linspace(*BOUNDS[k], POINTS_PER_DIM)),
        tolerance=0.01,
    )
    for k in range(DIMENSION)
]

searchspace = SearchSpace.from_product(parameters=parameters)
objective = SingleTargetObjective(target=NumericalTarget(name="Target", mode="MIN"))

### Constructing campaigns for the simulation loop

# To simplify adjusting the example for other recommenders, we construct some recommender objects.
# For details on recommender objects, we refer to [`recommenders`](./../Basics/recommenders.md).

seq_greedy_EI_recommender = TwoPhaseMetaRecommender(
    recommender=BotorchRecommender(acquisition_function="qEI"),
)
random_recommender = TwoPhaseMetaRecommender(recommender=RandomRecommender())

# We now create one campaign per recommender.

seq_greedy_EI_campaign = Campaign(
    searchspace=searchspace,
    recommender=seq_greedy_EI_recommender,
    objective=objective,
)
random_campaign = Campaign(
    searchspace=searchspace,
    recommender=random_recommender,
    objective=objective,
)

### Performing the simulation loop

# We can now use the `simulate_scenarios` function to simulate a full experiment.
# Note that this function enables to run multiple scenarios by a single function call.
# For this, it is necessary to define a dictionary mapping scenario names to campaigns.

scenarios = {
    "Sequential greedy EI": seq_greedy_EI_campaign,
    "Random": random_campaign,
}
results = simulate_scenarios(
    scenarios,
    sum_of_squares,
    batch_size=3,
    n_doe_iterations=N_DOE_ITERATIONS,
    n_mc_iterations=N_MC_ITERATIONS,
)

# The following lines plot the results and save the plot in run_analytical.png

sns.lineplot(data=results, x="Num_Experiments", y="Target_CumBest", hue="Scenario")
plt.gcf().set_size_inches(24, 8)
plt.savefig("./run_analytical.png")
