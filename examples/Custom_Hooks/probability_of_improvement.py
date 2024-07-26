## Monitoring the Probability of Improvement

# This example demonstrates how the {func}`register_hooks <baybe.utils.basic.register_hooks>` utility can be used to extract the probability of improvement (PI) from a running campaign:
# * We define a hook that is compatible with the {meth}`BotorchRecommender.recommend <baybe.recommenders.pure.bayesian.BotorchRecommender.recommend>` interface and lets us extract the PI achieved after each experimental iteration,
# * attach the hook to the recommender driving our campaign,
# * and plot the evolving PI values after campaign completion.


### Imports

import os
from types import MethodType

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from botorch.test_functions.synthetic import Hartmann
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde

from baybe.acquisition import ProbabilityOfImprovement
from baybe.campaign import Campaign
from baybe.objectives.base import Objective
from baybe.objectives.single import SingleTargetObjective
from baybe.parameters import NumericalDiscreteParameter
from baybe.recommenders import (
    BotorchRecommender,
    RandomRecommender,
    TwoPhaseMetaRecommender,
)
from baybe.searchspace import SearchSpace, SearchSpaceType
from baybe.surrogates import GaussianProcessSurrogate
from baybe.targets import NumericalTarget
from baybe.utils.basic import register_hooks
from baybe.utils.botorch_wrapper import botorch_function_wrapper
from baybe.utils.dataframe import to_tensor
from baybe.utils.plotting import create_example_plots
from baybe.utils.random import set_random_seed

### Settings for a full simulation loop

# For the simulation, we need to define some basic settings like the number of
# iterations or the batch size:

SMOKE_TEST = "SMOKE_TEST" in os.environ
N_DOE_ITERATIONS = 3 if SMOKE_TEST else 7
BATCH_SIZE = 2
DIMENSION = 3
POINTS_PER_DIM = 2 if SMOKE_TEST else 4

# We also fix the random seed to create a consistent plot:

set_random_seed(1282)

### Setup

# We initialize a container for storing the PI from each iteration:

pi_per_iteration: list[np.ndarray] = []

# Then, we define the hook that calculates the PI.
# To attach the hook, we need to match its signature to that of {meth}`RecommenderProtocol.recommend <baybe.recommenders.base.RecommenderProtocol.recommend>`.


def extract_pi(
    self: BotorchRecommender,
    searchspace: SearchSpace,
    objective: Objective | None = None,
    measurements: pd.DataFrame | None = None,
) -> None:
    """Calculate and store the probability of improvement."""
    if searchspace.type is not SearchSpaceType.DISCRETE:
        raise TypeError(
            f"Search spaces of type '{searchspace.type}' are not supported. "
            f"Currently, only search spaces of type '{SearchSpaceType.DISCRETE}' are accepted."
        )
    train_x = searchspace.transform(measurements)
    train_y = objective.transform(measurements)
    acqf = ProbabilityOfImprovement()
    botorch_acqf = acqf.to_botorch(self.surrogate_model, searchspace, train_x, train_y)
    comp_rep_tensor = to_tensor(searchspace.discrete.comp_rep).unsqueeze(1)
    with torch.no_grad():
        pi = botorch_acqf(comp_rep_tensor)
    pi_per_iteration.append(pi.numpy())


# Additionally, we define a function that plots the collected PI values:


def create_pi_plot(
    pi_per_iteration: list[np.ndarray],
) -> Axes3D:
    """Create the plot of the probability of improvement in 3D."""
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    cmap = plt.get_cmap("viridis")
    pi_max = max([np.max(p) for p in pi_per_iteration])

    # Plot each PI array separately
    for i, p in enumerate(pi_per_iteration):
        x = np.linspace(0, pi_max, 500)
        kde = gaussian_kde(p)
        y = kde(x)
        z = np.full_like(y, i)

        # Fill the area under the curve
        verts = []
        verts.append([(x[i], 0.0), *zip(x, y), (x[-1], 0.0)])
        color = cmap(float(i) / len(pi_per_iteration))
        poly = PolyCollection(verts, color=color, alpha=0.9)
        ax.add_collection3d(poly, zs=i, zdir="x")

        ax.plot(x, y, z, zdir="x", color=color)

    # Set viewing angle
    ax.view_init(elev=20, azim=30)

    # Reduce space between the iterations
    ax.set_box_aspect([0.7, 1, 1])

    # Set the axis limit based on the maximal PI
    ax.set_ylim(0, pi_max)

    # Set axis ticks to have the correct iteration number
    ax.set_xticks(np.arange(0, len(pi_per_iteration), 1))
    ax.set_xticklabels([i for i in range(1, len(pi_per_iteration) + 1)])
    ax.set_ylabel("PI", labelpad=20)
    ax.set_xlabel("Iteration", labelpad=20)
    ax.set_zlabel("Density", labelpad=20)

    return ax


### Monkeypatching

# Next, we create our recommender and monkeypatch its `recommend` method:

bayesian_recommender = BotorchRecommender(
    surrogate_model=GaussianProcessSurrogate(),
)
bayesian_recommender.recommend = MethodType(
    register_hooks(
        BotorchRecommender.recommend,
        post_hooks=[extract_pi],
    ),
    bayesian_recommender,
)
recommender = TwoPhaseMetaRecommender(
    initial_recommender=RandomRecommender(),
    recommender=bayesian_recommender,
)


# In this example, we use `MethodType` to bind the `BotorchRecommender.recommend`
# **function** with our hook.
# For more information, we refer to the [`basic hook example`](./basics.md).


### Triggering the Hook

# We set up the campaign:

test_function = Hartmann(dim=DIMENSION)
WRAPPED_FUNCTION = botorch_function_wrapper(test_function=test_function)
BOUNDS = test_function.bounds

discrete_params = [
    NumericalDiscreteParameter(
        name=f"x{d}",
        values=np.linspace(lower, upper, POINTS_PER_DIM),
    )
    for d, (lower, upper) in enumerate(BOUNDS.T)
]

searchspace = SearchSpace.from_product(parameters=discrete_params)
objective = SingleTargetObjective(target=NumericalTarget(name="Target", mode="MIN"))
campaign = Campaign(
    searchspace=searchspace,
    recommender=recommender,
    objective=objective,
)

# Now, we perform a couple of iterations:

for i in range(N_DOE_ITERATIONS):
    recommendation = campaign.recommend(BATCH_SIZE)
    target_values = recommendation.apply(WRAPPED_FUNCTION, axis=1)
    recommendation["Target"] = target_values
    campaign.add_measurements(recommendation)

# Lastly, we plot the PI from the previous iterations:

ax = create_pi_plot(pi_per_iteration)
create_example_plots(ax=ax, base_name="probability_of_improvement")
