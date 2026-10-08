# # Optimizing a Permutation-Invariant Function

# In this example, we explore BayBE's capabilities for handling optimization problems
# with symmetry. We compare three ways of exploiting a permutation invariance:
# restricting the search space via a constraint, augmenting the measurements, and
# building the invariance directly into the kernel of a Gaussian process.

# ## Imports

import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from baybe import Campaign, Settings
from baybe.constraints import DiscretePermutationInvarianceConstraint
from baybe.parameters import NumericalDiscreteParameter
from baybe.recommenders import (
    BotorchRecommender,
    TwoPhaseMetaRecommender,
)
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.surrogates import GaussianProcessSurrogate, NGBoostSurrogate
from baybe.targets import NumericalTarget

# ## Settings

Settings(random_seed=1337).activate()
SMOKE_TEST = "SMOKE_TEST" in os.environ
N_MC_ITERATIONS = 2 if SMOKE_TEST else 100
N_DOE_ITERATIONS = 2 if SMOKE_TEST else 50

# ## The Scenario

# We will explore a 2-dimensional permutation-invariant function, i.e., it holds that
# $f(x,y) = f(y,x)$. The function was crafted to exhibit no additional mirror symmetry
# (a common way of also resulting in permutation invariance) and have multiple minima.
# In practice, permutation invariance can arise e.g. for
# [mixtures when modeled with a slot-based approach](/examples/Mixtures/slot_based).

LBOUND = -2.0
UBOUND = 2.0


def lookup(df: pd.DataFrame) -> pd.DataFrame:
    """A lookup modeling a permutation-invariant 2D function with multiple minima."""
    x = df["x"].values
    y = df["y"].values
    result = (
        (x - y) ** 2
        + (x**3 + y**3)
        + ((x**2 - 1) ** 2 + (y**2 - 1) ** 2)
        + np.sin(3 * (x + y)) ** 2
        + np.sin(3 * np.abs(x - y)) ** 2
    )

    df_z = pd.DataFrame({"f": result}, index=df.index)
    return df_z


x = np.linspace(LBOUND, UBOUND, 25)
y = np.linspace(LBOUND, UBOUND, 25)
xx, yy = np.meshgrid(x, y)
df_plot = lookup(pd.DataFrame({"x": xx.ravel(), "y": yy.ravel()}))
zz = df_plot["f"].values.reshape(xx.shape)
line_vals = np.linspace(LBOUND, UBOUND, 2)

# fmt: off
fig, ax = plt.subplots(figsize=(7.5, 6))
contour = ax.contourf(xx, yy, zz, levels=50, cmap="viridis")
fig.colorbar(contour, ax=ax)
ax.plot(line_vals, line_vals, "r--", alpha=0.5, linewidth=2)
ax.set_title("Ground Truth: $f(x, y)$ = $f(y, x)$ (Permutation Invariant)")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.tight_layout()
plt.show()
# fmt: on

# The plot shows the function we want to minimize. The dashed red line illustrates the
# permutation invariance, which is similar to a mirror symmetry, just not along any of
# the parameter axes but along the diagonal. We can also see several local minima.

# Such a situation can be challenging for optimization algorithms if no information
# about the invariance is considered. BayBE offers several ways to account for it:

# - **Constraint:** If no
#   {class}`~baybe.constraints.discrete.DiscretePermutationInvarianceConstraint` is
#   used, BayBE searches for the optima across the entire 2D space. However, the search
#   can be restricted to the lower (or equivalently the upper) triangle of the search
#   space, which is exactly what the constraint does: It removes entries that are
#   "duplicated" in the sense of already being represented by another invariant point.
# - **Data augmentation:** If a recommender is configured with
#   {attr}`~baybe.recommenders.pure.bayesian.base.BayesianRecommender.symmetries`, the
#   model is fit with an extended set of points, including the permuted versions of all
#   measurements. Depending on the surrogate model, this might have different impacts.
#   For example, we can expect a strong effect for tree-based models like the
#   {class}`~baybe.surrogates.ngboost.NGBoostSurrogate`, because their splits are
#   always parallel to the parameter axes. Without augmented measurements, it is thus
#   easy to fall into suboptimal splits and overfit.
# - **Invariant kernels:** If a
#   {class}`~baybe.surrogates.gaussian_process.core.GaussianProcessSurrogate` is
#   configured with
#   {attr}`~baybe.surrogates.gaussian_process.core.GaussianProcessSurrogate.symmetries`,
#   its kernel is constructed such that its predictions are exactly invariant, without
#   requiring additional training points.

# ## The Optimization Problem

p1 = NumericalDiscreteParameter("x", np.linspace(LBOUND, UBOUND, 51))
p2 = NumericalDiscreteParameter("y", np.linspace(LBOUND, UBOUND, 51))
objective = NumericalTarget("f", minimize=True).to_objective()

# We set up a constrained and an unconstrained search space to demonstrate the impact
# of the constraint on optimization performance.

constraint = DiscretePermutationInvarianceConstraint(["x", "y"])
searchspace_plain = SearchSpace.from_product([p1, p2])
searchspace_constrained = SearchSpace.from_product([p1, p2], [constraint])

print("Number of Points in the Search Space")
print(f"{'Without Constraint:':<35} {len(searchspace_plain.discrete.exp_rep)}")
print(f"{'With Constraint:':<35} {len(searchspace_constrained.discrete.exp_rep)}")

# We can see that the unconstrained search space has roughly twice as many points
# compared to the constrained one. This is expected, as the
# {class}`~baybe.constraints.discrete.DiscretePermutationInvarianceConstraint`
# effectively models only one half of the parameter triangle. Note that the factor is
# not exactly 2 due to the (still included) points on the diagonal.

# To construct the corresponding symmetry conveniently, we use the `to_symmetry` method
# of the constraint. We then compare four models: an NGBoost model and a Gaussian
# process, each without exploiting the symmetry, the NGBoost model with data
# augmentation, and the Gaussian process with an invariant kernel.

symmetry = constraint.to_symmetry()
surrogates = {
    "NGBoost": NGBoostSurrogate(),
    "NGBoost, augmented": NGBoostSurrogate(),
    "GP": GaussianProcessSurrogate(),
    "GP, invariant kernel": GaussianProcessSurrogate(symmetries=[symmetry]),
}


def make_recommender(model: str) -> TwoPhaseMetaRecommender:
    """Create a recommender for the given model, augmenting data where requested."""
    return TwoPhaseMetaRecommender(
        recommender=BotorchRecommender(
            surrogate_model=surrogates[model],
            symmetries=[symmetry] if model.endswith("augmented") else [],
        )
    )


# Combining all models with both search spaces results in eight campaigns:

searchspaces = {"No": searchspace_plain, "Yes": searchspace_constrained}
scenarios = {
    f"{model}|{constrained}": Campaign(searchspace, objective, make_recommender(model))
    for model in surrogates
    for constrained, searchspace in searchspaces.items()
}

# ## Simulating the Optimization Loop

results = simulate_scenarios(
    scenarios,
    lookup,
    n_doe_iterations=N_DOE_ITERATIONS,
    n_mc_iterations=N_MC_ITERATIONS,
).rename(
    columns={
        "f_CumBest": "$f(x,y)$ (cumulative best)",
        "Num_Experiments": "# Experiments",
    }
)
results[["Model", "Constrained"]] = results["Scenario"].str.split("|", expand=True)

# ## Results

# Let us visualize the optimization process, with and without the constraint:

fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
for ax, (constrained, title) in zip(
    axs, [("No", "Without Constraint"), ("Yes", "With Constraint")]
):
    sns.lineplot(
        data=results[results["Constrained"] == constrained],
        x="# Experiments",
        y="$f(x,y)$ (cumulative best)",
        hue="Model",
        hue_order=list(surrogates),
        marker="o",
        markevery=2,
        markersize=8,
        ax=ax,
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(ax.get_ylim()[0], 3)
    ax.set_title(title)
plt.tight_layout()
plt.show()

# Each panel compares the four models on one of the search spaces. Comparing the
# curves within a panel shows the effect of data augmentation and invariant kernels,
# while comparing the two panels shows the effect of the constraint.
