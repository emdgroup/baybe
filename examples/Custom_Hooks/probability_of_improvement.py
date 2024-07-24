## Example for calculating and plotting the Probability of Improvement

# This example demonstrates how the {func}`register_hooks <baybe.utils.basic.register_hooks>` utility could be used to extract the probability of improvement (PI) from a running campaign:
# * We define a hook that is compatible with the {meth}`BotorchRecommender.recommend <baybe.recommenders.pure.bayesian.BotorchRecommender.recommend>` interface and let's us extract the PI achieved after each experimental iteration,
# * attach the hook to the recommender driving our campaign,
# * and plot the evolving PI values after campaign completion.


### Imports

import os
import warnings
from pathlib import Path
from types import MethodType

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from botorch.test_functions.synthetic import Hartmann
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
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
from baybe.utils.random import set_random_seed

### Parameters for a full simulation loop

# For the full simulation, we need to define some parameters.

SMOKE_TEST = "SMOKE_TEST" in os.environ
N_DOE_ITERATIONS = 3 if SMOKE_TEST else 7
BATCH_SIZE = 2
DIMENSION = 3
POINTS_PER_DIM = 2 if SMOKE_TEST else 4

# We also define the random seed to avoid having different plots.

set_random_seed(1282)

### Setup

# We initialize a container for storing the PI from each recommend iteration:

pi_per_iteration: list[np.ndarray] = []

# Then, we define the hook that calculates the PI from each iteration.
# To attach the hook we need to match its signature to that of {meth}RecommenderProtocol.recommend <baybe.recommenders.base.RecommenderProtocol.recommend>.


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


# Additionally, we define a function that plots the PI after all recommend iterations:


def plot_pi(
    pi_per_iteration: list[np.ndarray],
    ax: Axes,
    base_name: str,
    path: Path = Path("."),
) -> Figure:
    """Plot the probability of improvement in 3D."""
    cmap = plt.get_cmap("viridis")
    pi_max = max([np.max(p) for p in pi_per_iteration])

    # Plot each PI tensor separately
    for i, p in enumerate(pi_per_iteration):
        x = np.linspace(0, pi_max, 500)
        kde = gaussian_kde(p)
        y = kde(x)
        z = np.full_like(y, i)

        # Fill under the curve
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

    # Set the y-axis limit based on the maximal PI
    ax.set_ylim(0, pi_max)

    # Set x-axis ticks to have the correct iteration number
    ax.set_xticks(np.arange(0, len(pi_per_iteration), 1))
    ax.set_xticklabels([i for i in range(1, len(pi_per_iteration) + 1)])

    ax.set_ylabel("PI")
    ax.set_xlabel("Iteration")
    ax.set_zlabel("Density")

    output_path = Path(path, base_name)
    if isinstance(ax.figure, Figure):
        ax.figure.savefig(
            output_path,
            format="svg",
            transparent=True,
        )
    else:
        warnings.warn("Plots could not be saved.")
    plt.close()
    return ax.get_figure()


### Monkeypatching

# Next, we create our recommender and monkeypatch its `recommend` method:

my_recommender = TwoPhaseMetaRecommender(
    initial_recommender=RandomRecommender(),
    recommender=BotorchRecommender(
        surrogate_model=GaussianProcessSurrogate(),
        allow_repeated_recommendations=True,
        allow_recommending_already_measured=True,
    ),
)
my_recommender.recommender.recommend = MethodType(
    register_hooks(
        BotorchRecommender.recommend,
        post_hooks=[extract_pi],
    ),
    my_recommender.recommender,
)

# In this example we use `MethodType` to bind the `BotorchRecommender.recommend`
# **function** with our hook.
# For more information, we refer to the [`basics example`](./basics.md).


### Triggering the Hook

# We setup the other objects to trigger the hook:

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
    recommender=my_recommender,
    objective=objective,
)

# Now we perform a couple of recommendations:

for i in range(N_DOE_ITERATIONS):
    recommendation = campaign.recommend(BATCH_SIZE)
    target_values = recommendation.apply(WRAPPED_FUNCTION, axis=1)
    recommendation["Target"] = target_values
    campaign.add_measurements(recommendation)

# Lastly, we plot the PI from the previous iterations:

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
plot_pi(pi_per_iteration, ax=ax, base_name="PI_Plot.svg")
