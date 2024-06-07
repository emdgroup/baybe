## Example for full simulation loop using a custom analytical test function in a hybrid space

# This example shows a simulation loop for a single target with a custom test function as lookup.
# Most importantly, it demonstrates the creation of a custom hybrid searchspace.

# This examples assumes some basic familiarity with using BayBE and the lookup mechanism.
# We refer to [`campaign`](./../Basics/campaign.md) for a more  basic example resp.
# to [`custom_analytical`](./custom_analytical.md) for details on the lookup mechanism.

### Necessary imports for this example

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from baybe import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalContinuousParameter, NumericalDiscreteParameter
from baybe.recommenders import (
    BotorchRecommender,
    NaiveHybridSpaceRecommender,
    RandomRecommender,
    TwoPhaseMetaRecommender,
)
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget

### Parameters for a full simulation loop

# For the full simulation, we need to define some additional parameters.
# These are the number of Monte Carlo runs and the number of experiments to be conducted per run.
# `POINTS_PER_DIM` denotes how many points each discrete dimension should contain.

SMOKE_TEST = "SMOKE_TEST" in os.environ

N_MC_ITERATIONS = 2 if SMOKE_TEST else 5
N_DOE_ITERATIONS = 2 if SMOKE_TEST else 5
POINTS_PER_DIM = 3 if SMOKE_TEST else 6


### Defining the test function.


# See [`here`](./custom_analytical.md) for details on the custom analytical test function.


def sum_of_squares(*x: float) -> float:
    """Calculate the sum of squares."""
    res = 0
    for y in x:
        res += y**2
    return res


# For our actual experiment, we need to specify the number of dimension that we want to use.
# This is necessary to know for the creation of the parameters.
# Similarly, it is necessary to state the bounds of the parameters.
# These should be provided as a list of two-dimensional tuples.

DIMENSION = 4
BOUNDS = [(-2, 2), (-2, 2), (-2, 2), (-2, 2)]

### Constructing the hybrid searchspace

# Our goal is to construct a hybrid searchspace containing discrete and continuous parameters.
# We thus need to specify which indices should be discrete and which should be continuous.

CONT_INDICES = [0, 1]
DISC_INDICES = [2, 3]

# This code verifies whether the provided indices agree with `DIMENSION`.

if set(CONT_INDICES + DISC_INDICES) != set(range(DIMENSION)):
    raise ValueError(
        "Either the intersection between CONT_IND and DISC_IND is not empty or your "
        "indices do not match."
    )


# Construct the continuous parameters as NumericContinuous parameters.

cont_parameters = [
    NumericalContinuousParameter(
        name=f"x_{k+1}",
        bounds=(BOUNDS[k]),
    )
    for k in CONT_INDICES
]

# Construct the discrete parameters as `NumericalDiscreteParameters`.

disc_parameters = [
    NumericalDiscreteParameter(
        name=f"x_{k+1}",
        values=list(np.linspace(*BOUNDS[k], POINTS_PER_DIM)),
        tolerance=0.01,
    )
    for k in DISC_INDICES
]

# Concatenate the continuous and discrete parameters.

parameters = cont_parameters + disc_parameters

# Construct searchspace and objective.
searchspace = SearchSpace.from_product(parameters=parameters)
objective = SingleTargetObjective(target=NumericalTarget(name="Target", mode="MIN"))

### Constructing campaigns for the simulation loop

# This example compares three different available hybrid recommenders:
# The `BotorchRecommender`, the `NaiveHybridSpaceRecommender` and the `RandomRecommender`.
# For each of them, we initialize one recommender object.
# Note that it is possible to further specify the behavior of the `BotorchRecommender`.
# Using the two keywords `hybrid_sampler` and `sampling_percentage`, one can control
# - how much of the discrete subspace should be explored
# - how these points should be sampled.

# Note that the recommender performs one optimization of the continuous subspace per sampled point.
# We thus recommend to keep this parameter rather low.

seq_greedy_recommender = TwoPhaseMetaRecommender(
    recommender=BotorchRecommender(hybrid_sampler="FPS", sampling_percentage=0.3),
)
naive_hybrid_recommender = TwoPhaseMetaRecommender(
    recommender=NaiveHybridSpaceRecommender()
)
random_recommender = TwoPhaseMetaRecommender(recommender=RandomRecommender())

# We now create one campaign per recommender.

seq_greedy_campaign = Campaign(
    searchspace=searchspace,
    recommender=seq_greedy_recommender,
    objective=objective,
)
naive_hybrid_campaign = Campaign(
    searchspace=searchspace,
    recommender=naive_hybrid_recommender,
    objective=objective,
)
random_campaign = Campaign(
    searchspace=searchspace,
    recommender=random_recommender,
    objective=objective,
)

# We can now use the `simulate_scenarios` function to simulate a full experiment.
# Note that this function enables to run multiple scenarios by a single function call.
# For this, it is necessary to define a dictionary mapping scenario names to campaigns.

scenarios = {
    "Sequential greedy": seq_greedy_campaign,
    "Naive hybrid": naive_hybrid_campaign,
    "Random": random_campaign,
}
results = simulate_scenarios(
    scenarios,
    sum_of_squares,
    batch_size=2,
    n_doe_iterations=N_DOE_ITERATIONS,
    n_mc_iterations=N_MC_ITERATIONS,
)

# The following lines plot the results and save the plot in run_analytical.png

sns.lineplot(data=results, x="Num_Experiments", y="Target_CumBest", hue="Scenario")
plt.gcf().set_size_inches(24, 8)
plt.savefig("./run_hybrid.png")
