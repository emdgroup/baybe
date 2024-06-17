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


import sys
import warnings
from pathlib import Path
from types import MethodType

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde

from baybe.acquisition import ProbabilityOfImprovement
from baybe.campaign import Campaign
from baybe.objectives.base import Objective
from baybe.objectives.single import SingleTargetObjective
from baybe.parameters import CategoricalParameter, NumericalDiscreteParameter
from baybe.recommenders import (
    BotorchRecommender,
    RandomRecommender,
    TwoPhaseMetaRecommender,
)
from baybe.searchspace import SearchSpace, SearchSpaceType
from baybe.surrogates import GaussianProcessSurrogate
from baybe.targets import NumericalTarget
from baybe.utils.basic import register_hooks
from baybe.utils.dataframe import add_fake_results, to_tensor

### Setup

# We start by initializing the global `poi_list` where we will store the PI from each recommend iteration:

poi_list: list[torch.Tensor] = []


# Then, we define the hook that calculates the PI based on the given recommender, search space and measurements:


def check_probability_of_improvement(
    self: BotorchRecommender,
    searchspace: SearchSpace,
    objective: Objective | None = None,
    measurements: pd.DataFrame | None = None,
):
    """Calculate and store the probability of improvement in poi_list."""
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
    poi = botorch_acqf(comp_rep_tensor)
    poi_list.append(poi)


# Additionally, we define a function that plots the `poi_list` after all recommend  iterations:


def plot_poi(poi_list: list[torch.Tensor]) -> None:
    """Plot the probability of improvement in 3D."""
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    cmap = plt.get_cmap("viridis")

    # Plot each PI tensor separately
    for i, p in enumerate(poi_list):
        poi_np = p.detach().numpy() if p.requires_grad else p.numpy()
        x = np.unique(poi_np)
        kde = gaussian_kde(poi_np)
        y = kde(x)
        z = np.full_like(y, i)

        # Fill under the curve
        verts = []
        verts.append([(x[i], 0.0), *zip(x, y), (x[-1], 0.0)])
        color = cmap(float(i) / len(poi_list))
        poly = PolyCollection(verts, color=color, alpha=0.9)
        ax.add_collection3d(poly, zs=i, zdir="x")

        ax.plot(x, y, z, zdir="x", color=color)

    # Set viewing angle
    ax.view_init(elev=20, azim=30)

    # Reduce space between the iterations
    ax.set_box_aspect([0.5, 1, 1])

    # Set the y-axis limit based on the maximal PI
    y_max = max([p.max().item() for p in poi_list])
    ax.set_ylim(0, y_max)

    ax.set_ylabel("PI")
    ax.set_xlabel("Iteration")
    ax.set_zlabel("Density")
    ax.set_title("Investigate the PI")

    output_path = Path(Path(sys.path[0]), "POI_Plot.svg")
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

dict_solvent = {
    "DMAc": r"CC(N(C)C)=O",
    "Butyornitrile": r"CCCC#N",
    "Butyl Ester": r"CCCCOC(C)=O",
    "p-Xylene": r"CC1=CC=C(C)C=C1",
}
dict_base = {
    "Potassium acetate": r"O=C([O-])C.[K+]",
    "Potassium pivalate": r"O=C([O-])C(C)(C)C.[K+]",
    "Cesium acetate": r"O=C([O-])C.[Cs+]",
    "Cesium pivalate": r"O=C([O-])C(C)(C)C.[Cs+]",
}
dict_ligand = {
    "BrettPhos": r"CC(C)C1=CC(C(C)C)=C(C(C(C)C)=C1)C2=C(P(C3CCCCC3)C4CCCCC4)C(OC)="
    "CC=C2OC",
    "Di-tert-butylphenylphosphine": r"CC(C)(C)P(C1=CC=CC=C1)C(C)(C)C",
    "(t-Bu)PhCPhos": r"CN(C)C1=CC=CC(N(C)C)=C1C2=CC=CC=C2P(C(C)(C)C)C3=CC=CC=C3",
    "Tricyclohexylphosphine": r"P(C1CCCCC1)(C2CCCCC2)C3CCCCC3",
    "PPh3": r"P(C1=CC=CC=C1)(C2=CC=CC=C2)C3=CC=CC=C3",
    "XPhos": r"CC(C1=C(C2=CC=CC=C2P(C3CCCCC3)C4CCCCC4)C(C(C)C)=CC(C(C)C)=C1)C",
    "P(2-furyl)3": r"P(C1=CC=CO1)(C2=CC=CO2)C3=CC=CO3",
    "Methyldiphenylphosphine": r"CP(C1=CC=CC=C1)C2=CC=CC=C2",
    "1268824-69-6": r"CC(OC1=C(P(C2CCCCC2)C3CCCCC3)C(OC(C)C)=CC=C1)C",
    "JackiePhos": r"FC(F)(F)C1=CC(P(C2=C(C3=C(C(C)C)C=C(C(C)C)C=C3C(C)C)C(OC)=CC=C2OC)"
    r"C4=CC(C(F)(F)F)=CC(C(F)(F)F)=C4)=CC(C(F)(F)F)=C1",
    "SCHEMBL15068049": r"C[C@]1(O2)O[C@](C[C@]2(C)P3C4=CC=CC=C4)(C)O[C@]3(C)C1",
    "Me2PPh": r"CP(C)C1=CC=CC=C1",
}
parameters = [
    CategoricalParameter(name="Solvent", values=dict_solvent.keys(), encoding="OHE"),
    CategoricalParameter(name="Base", values=dict_base.keys(), encoding="OHE"),
    CategoricalParameter(name="Ligand", values=dict_ligand.keys(), encoding="OHE"),
    NumericalDiscreteParameter(name="Temp_C", values=[90, 105, 120], tolerance=2),
    NumericalDiscreteParameter(
        name="Concentration", values=[0.057, 0.1, 0.153], tolerance=0.005
    ),
]
searchspace = SearchSpace.from_product(parameters=parameters)
objective = SingleTargetObjective(target=NumericalTarget(name="yield", mode="MAX"))
campaign = Campaign(
    searchspace=searchspace,
    recommender=my_recommender,
    objective=objective,
)

# Now we perform 5 recommendations:

for i in range(7):
    recommendation = campaign.recommend(batch_size=3)
    add_fake_results(recommendation, campaign)
    campaign.add_measurements(recommendation)

# Lastly, we plot the PI from the previous iterations to be able to analyse them:

plot_poi(poi_list)
