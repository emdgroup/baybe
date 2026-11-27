# # Comparing Transfer Learning Modes on Historic Campaigns of Unknown Relevance

# In this example, we compare different
# {doc}`transfer learning modes </concepts/transfer_learning>` on historic campaigns
# whose relevance for the current campaign is unknown. We show that:
# * optimizing without historic data easily gets stuck in a local optimum,
# * pooling all data into a single model is misled by an unrelated campaign,
# * assuming positively correlated campaigns requires several measurements of the
#   current campaign to overcome the bias introduced by an unrelated campaign,
# * the rank-weighted ensemble identifies the useful campaign from just a few
#   measurements of the current campaign.

# ## Imports

import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from baybe import Campaign, active_settings
from baybe.parameters import NumericalDiscreteParameter, TaskParameter
from baybe.parameters.enum import TransferLearningMode
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget
from examples.utils import create_example_plots

# ## Settings

active_settings.random_seed = 1337

SMOKE_TEST = "SMOKE_TEST" in os.environ
N_MC_ITERATIONS = 2 if SMOKE_TEST else 30
N_DOE_ITERATIONS = 2 if SMOKE_TEST else 12
N_SOURCE_POINTS = 25  # number of measurements available per historic campaign

# ## The Scenario

# Imagine you want to find the reaction temperature that maximizes the yield of a new
# substrate. Your lab has run similar optimizations before and you have data from two
# historic campaigns at hand:
# * **Similar**: a closely related substrate whose yield responds to temperature
#   almost in the same way, with optima shifted to slightly higher temperatures.
# * **Decoy**: a substrate whose yield peaks exactly where the new substrate performs
#   worst.

# Which of these campaigns are helpful is not known in advance.

temperatures = np.linspace(0, 1, 101)


def peak(center: float, width: float) -> np.ndarray:
    """Return a Gaussian-shaped peak on the temperature grid."""
    return np.exp(-((temperatures - center) ** 2) / (2 * width**2))


true_yields = {
    "Target": 0.7 * peak(0.25, 0.12) + peak(0.75, 0.05),
    "Similar": 0.7 * peak(0.27, 0.12) + peak(0.77, 0.05),
    "Decoy": 0.7 * peak(0.59, 0.06),
}
rng = np.random.default_rng(0)
yields = {
    name: np.clip(values + rng.normal(0, 0.05, len(temperatures)), 0, 1)
    for name, values in true_yields.items()
}

# The current substrate has a narrow global optimum at a high temperature and a broad
# local optimum at a lower temperature, which makes it easy to get stuck.

# ## Lookup Tables

# We store the yield curves in lookup tables. The table of the current substrate closes
# the optimization loop, while random subsets of the historic tables serve as the
# available historic data.

lookups = {
    name: pd.DataFrame({"Temperature": temperatures, "Yield": values, "Task": name})
    for name, values in yields.items()
}
lookup_target = lookups.pop("Target")
lookup_sources = pd.concat(lookups.values())

# In each Monte Carlo run, a different subset of the historic data is available:

initial_data = [
    lookup_sources.groupby("Task").sample(N_SOURCE_POINTS, random_state=i)
    for i in range(N_MC_ITERATIONS)
]

# ## Campaigns

# The {class}`~baybe.parameters.categorical.TaskParameter` distinguishes the campaigns.
# Recommendations are only requested for the current substrate:

temperature = NumericalDiscreteParameter("Temperature", temperatures)
target = NumericalTarget("Yield")


def make_campaign(mode: TransferLearningMode) -> Campaign:
    """Create a campaign using the given transfer learning mode."""
    task = TaskParameter(
        "Task",
        values=["Target", *lookups],
        active_values=["Target"],
        override_transfer_learning_mode=mode,
    )
    return Campaign(SearchSpace.from_product([temperature, task]), target)


modes = [
    TransferLearningMode.NAIVE,
    TransferLearningMode.POSITIVE_INDEX_KERNEL,
    TransferLearningMode.RGPE,
]
scenarios = {mode.name: make_campaign(mode) for mode in modes}

# ## Simulation

# We run the transfer learning modes with the historic data and, as a baseline, a
# campaign without any historic data:

results_tl = simulate_scenarios(
    scenarios,
    lookup_target,
    initial_data=initial_data,
    batch_size=1,
    n_doe_iterations=N_DOE_ITERATIONS,
)
results_baseline = simulate_scenarios(
    {"No transfer learning": Campaign(SearchSpace.from_product([temperature]), target)},
    lookup_target.drop(columns="Task"),
    batch_size=1,
    n_doe_iterations=N_DOE_ITERATIONS,
    n_mc_iterations=N_MC_ITERATIONS,
)
results = pd.concat([results_tl, results_baseline])

# ## Results

# The plot shows the best yield found so far, averaged over the Monte Carlo runs:

fig, ax = plt.subplots()
sns.lineplot(
    data=results,
    marker="o",
    markersize=10,
    x="Num_Experiments",
    y="Yield_CumBest",
    hue="Scenario",
    ax=ax,
)
create_example_plots(ax=ax, base_name="transfer_learning_modes")

# * Without historic data, the optimization often settles in the broad local optimum
#   at low temperatures and rarely discovers the narrow global optimum.
# * {attr}`~baybe.parameters.enum.TransferLearningMode.NAIVE` pools all campaigns into a
#   single model, which treats the decoy's peak as if it belonged to the current
#   substrate and blurs the valley next to the global optimum.
# * {attr}`~baybe.parameters.enum.TransferLearningMode.POSITIVE_INDEX_KERNEL` assumes
#   that all campaigns are positively correlated. This does not hold for the decoy,
#   whose yield peaks where the current substrate performs worst. The model needs
#   several measurements of the current substrate to overcome this positive
#   correlation bias.
# * {attr}`~baybe.parameters.enum.TransferLearningMode.RGPE` fits one Gaussian process
#   per campaign and weights them by how well they rank the measurements of the current
#   substrate. This ranking quickly reveals that the similar campaign is a useful proxy
#   while the decoy is not, so RGPE finds the global optimum early in this scenario.
