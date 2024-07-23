## Example for calculating and plotting the Probability of Improvement

# This example demonstrates how to use the
# {func}`register_hooks <baybe.utils.basic.register_hooks>` utility
# to calculate and analyse the probability of improvement (PI) evolving during iterations:
# * We define a hook that is compatible with the general
#   {meth}`RecommenderProtocol.recommend <baybe.recommenders.base.RecommenderProtocol.recommend>`
#   interface,
# * attach it to a recommender,
# * and watch it take action.


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

# For the full simulation, we need to define some additional parameters. These are the number of experiments to be conducted per run, the batch size, the dimension and the points per dimension.

SMOKE_TEST = "SMOKE_TEST" in os.environ
N_DOE_ITERATIONS = 3 if SMOKE_TEST else 7
BATCH_SIZE = 2
DIMENSION = 3
POINTS_PER_DIM = 4

# We define the random seed to avoid having different plots.
set_random_seed(1282)

### Setup

# We start by initializing the global `poi_list` where we will store the PI from each recommend iteration:

pi_list: list[torch.Tensor] = []


# Then, we define the hook that calculates the PI based on the given recommender, search space and measurements:


def check_probability_of_improvement(
    self: BotorchRecommender,
    searchspace: SearchSpace,
    objective: Objective | None = None,
    measurements: pd.DataFrame | None = None,
) -> list[torch.Tensor]:
    """Calculate and store the probability of improvement in poi_list.

    For reasons of numerical stability, the function adds some noise to the calculated
    probabilities.
    """
    if searchspace.type != SearchSpaceType.DISCRETE:
        raise TypeError(
            f"{searchspace.type} search spaces are not supported yet. "
            f"Currently only DISCRETE search spaces are accepted."
        )
    train_x = searchspace.transform(measurements)
    train_y = objective.transform(measurements)
    acqf = ProbabilityOfImprovement()
    botorch_acqf = acqf.to_botorch(self.surrogate_model, searchspace, train_x, train_y)
    comp_rep_tensor = to_tensor(searchspace.discrete.comp_rep).unsqueeze(1)
    pi = botorch_acqf(comp_rep_tensor)
    pi = pi + 1e-10 * torch.randn(pi.shape)
    pi_list.append(pi)


# Additionally, we define a function that plots the `poi_list` after all recommend  iterations:


def plot_pi(
    pi_list: list[torch.Tensor],
    ax: Axes,
    base_name: str,
    path: Path = Path("."),
) -> Figure:
    """Plot the probability of improvement in 3D."""
    cmap = plt.get_cmap("viridis")

    pi_max = max([torch.max(p).item() for p in pi_list])

    # Plot each PI tensor separately
    for i, p in enumerate(pi_list):
        pi_np = p.detach().numpy() if p.requires_grad else p.numpy()
        x = np.linspace(0, pi_max, 500)
        kde = gaussian_kde(pi_np)
        y = kde(x)
        z = np.full_like(y, i)

        # Fill under the curve
        verts = []
        verts.append([(x[i], 0.0), *zip(x, y), (x[-1], 0.0)])
        color = cmap(float(i) / len(pi_list))
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
    ax.set_xticks(np.arange(0, len(pi_list), 1))
    ax.set_xticklabels([i for i in range(1, len(pi_list) + 1)])

    ax.set_ylabel("PI")
    ax.set_xlabel("Iteration")
    ax.set_zlabel("Density")
    ax.set_title("Investigate the PI")

    output_path = Path(path, base_name)
    # mypy thinks that ax.figure might become None, hence the explicit ignore
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
        post_hooks=[check_probability_of_improvement],
    ),
    my_recommender.recommender,
)

# In this example we use `MethodType` to bind the `RandomRecommender.recommend`
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

# Now we perform 7 recommendations:

for i in range(N_DOE_ITERATIONS):
    recommendation = campaign.recommend(BATCH_SIZE)
    target_values = recommendation.apply(WRAPPED_FUNCTION, axis=1)
    recommendation["Target"] = target_values
    campaign.add_measurements(recommendation)

# Lastly, we plot the PI from the previous iterations to be able to analyse them:

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
plot_pi(pi_list, ax=ax, base_name="PI_Plot.svg")
