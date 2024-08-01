## Campaign Stopping

# Based on the insights from
# [this other example](/examples/Custom_Hooks/probability_of_improvement), we now
# demonstrate how to leverage the
# {func}`register_hooks <baybe.utils.basic.register_hooks>`
# mechanics to interrupt a running campaign based on a simple *Probability of
# Improvement (PI)* criterion. This approach could be used, for instance, to terminate
# unpromising campaigns early and refine their search spaces, or to end an ongoing
# optimization if the found results are sufficiently good.

# The underlying use case is taken from the example shown
# [here](/examples/Backtesting/full_lookup).


### Imports

import os
import warnings

import pandas as pd
import seaborn as sns
import torch

from baybe import Campaign
from baybe.acquisition import ProbabilityOfImprovement
from baybe.exceptions import UnusedObjectWarning
from baybe.objectives import SingleTargetObjective
from baybe.objectives.base import Objective
from baybe.parameters import NumericalDiscreteParameter, SubstanceParameter
from baybe.recommenders import (
    BotorchRecommender,
    RandomRecommender,
    TwoPhaseMetaRecommender,
)
from baybe.searchspace import SearchSpace, SearchSpaceType
from baybe.simulation import simulate_experiment, simulate_scenarios
from baybe.targets import NumericalTarget
from baybe.utils import register_hooks
from baybe.utils.dataframe import to_tensor
from baybe.utils.plotting import create_example_plots
from baybe.utils.random import set_random_seed

### Temporary
warnings.filterwarnings(
    "ignore", category=UnusedObjectWarning, message="explicit objective"
)
warnings.filterwarnings("ignore", category=DeprecationWarning)


### Settings

# Let's start by defining some basic settings required for the example:

SMOKE_TEST = "SMOKE_TEST" in os.environ
N_DOE_ITERATIONS = 2 if SMOKE_TEST else 20
N_MC_ITERATIONS = 2 if SMOKE_TEST else 50
N_STOPPED_CAMPAIGNS = 2 if SMOKE_TEST else 5
BATCH_SIZE = 1

# We also fix the random seed for reproducibility:

set_random_seed(1337)


### Problem Definition and Lookup Functionality

# Following the setup described [here](../Backtesting/full_lookup.md), we create the
# building blocks for the optimization problem:

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
    SubstanceParameter(name="Solvent", data=dict_solvent, encoding="MORDRED"),
    SubstanceParameter(name="Base", data=dict_base, encoding="MORDRED"),
    SubstanceParameter(name="Ligand", data=dict_ligand, encoding="MORDRED"),
    NumericalDiscreteParameter(name="Temp_C", values=[90, 105, 120], tolerance=2),
    NumericalDiscreteParameter(name="Concentration", values=[0.057, 0.1, 0.153]),
]

objective = SingleTargetObjective(target=NumericalTarget(name="yield", mode="MAX"))

# Also, we load the dataframe containing the lookup data for the closed-loop simulation:

try:
    lookup = pd.read_excel("./../Backtesting/lookup.xlsx")
except FileNotFoundError:
    lookup = pd.read_excel("examples/Backtesting/lookup.xlsx")

### Simulating the Unstopped Campaigns

# First, we run several Monte Carlo repetitions of the uninterrupted campaign to get a
# feeling for the average trajectory:

scenarios = {
    "Average uninterrupted": Campaign(
        searchspace=SearchSpace.from_product(parameters=parameters), objective=objective
    )
}

results = simulate_scenarios(
    scenarios,
    lookup,
    batch_size=BATCH_SIZE,
    n_doe_iterations=N_DOE_ITERATIONS,
    n_mc_iterations=N_MC_ITERATIONS,
)


### Defining the Campaign-Stopping Hook

# In order to interrupt a running campaign, we define a custom exception to identify our
# stopping event:


class CampaignStoppedException(Exception):
    """The campaign should be stopped."""


# Based on this exception class, we can now define a hook implementing the stopping
# criterion. For this purpose, we count the fraction of candidates with a PI exceeding a
# given value and terminate the campaign once the fraction falls below a certain
# threshold.

PI_THRESHOLD = 0.01  # PI of 1% to identify promising points
PI_REQUIRED_FRACTION = 0.1  # 10% of candidates must be above the threshold


def stop_on_PI(
    self: BotorchRecommender,
    searchspace: SearchSpace,
    objective: Objective | None = None,
    measurements: pd.DataFrame | None = None,
) -> None:
    """Raise an exception if the PI-based stopping criterion is fulfilled."""
    if searchspace.type != SearchSpaceType.DISCRETE:
        raise TypeError(
            f"Search spaces of type '{searchspace.type}' are not supported. "
            f"Currently, only search spaces of type '{SearchSpaceType.DISCRETE}' are "
            f"accepted."
        )
    train_x = searchspace.transform(measurements)
    train_y = objective.transform(measurements)
    acqf = ProbabilityOfImprovement()

    botorch_acqf = acqf.to_botorch(self.surrogate_model, searchspace, train_x, train_y)
    _, candidates_comp_rep = searchspace.discrete.get_candidates(
        allow_repeated_recommendations=self.allow_repeated_recommendations,
        allow_recommending_already_measured=self.allow_recommending_already_measured,
    )
    comp_rep_tensor = to_tensor(candidates_comp_rep).unsqueeze(1)
    acqf_values = botorch_acqf(comp_rep_tensor)

    n_pis_over = torch.sum(acqf_values > PI_THRESHOLD)
    n_pis_over_required = len(candidates_comp_rep) * PI_REQUIRED_FRACTION
    if n_pis_over < n_pis_over_required:
        raise CampaignStoppedException(
            f"Less than {PI_REQUIRED_FRACTION*100:.0f}% of candidates are above the PI "
            f"threshold of {PI_THRESHOLD*100:.0f}% - Stopping the campaign."
        )


# Now, we attach the hook to the ``recommend`` function of our recommender class:

BotorchRecommender.recommend = register_hooks(
    BotorchRecommender.recommend, post_hooks=[stop_on_PI]
)

# ```{admonition} Monkeypatching
# :class: note
# The above monkeypatch registers the hook with all future instances of the recommender
# class. While it is possible to attach the hook only to a specific instance via
# ``MethodType`` (see [here](./basics.md)), this approach does not work well with the
# simulation utilities because they internally create deep copies of the simulated
# campaign, effectively bypassing the patch.
# ```

### Simulating the Stopped Campaigns

# With the hook attached, we again run several Monte Carlo repetitions of the same
# campaign:

stopped_results = []
for k in range(N_STOPPED_CAMPAIGNS):
    # Create a fresh campaign
    campaign = Campaign(
        searchspace=SearchSpace.from_product(parameters=parameters),
        objective=objective,
        recommender=TwoPhaseMetaRecommender(
            initial_recommender=RandomRecommender(),
            recommender=BotorchRecommender(),
        ),
    )

    # Run the simulation loop
    # (If an exception is thrown inside the loop, the function still returns the partial
    # trajectory, which effectively implements early stopping.)
    result = simulate_experiment(
        campaign,
        lookup,
        batch_size=BATCH_SIZE,
        n_doe_iterations=N_DOE_ITERATIONS,
    )

    # Contrary to ``simulate_scenarios``, ``simulate_experiment`` runs on single
    # iteration, and we have to add the scenario information ourselves here.
    result["Scenario"] = f"PI-stopped, run {k+1}"
    stopped_results.append(result)

### Plotting the Results

# Finally, we plot both the stopped and the unstopped trajectories:

results = pd.concat([results, *stopped_results])
ax = sns.lineplot(
    data=results,
    marker="o",
    markersize=10,
    x="Num_Experiments",
    y="yield_CumBest",
    hue="Scenario",
)
for line in ax.get_lines()[1:]:
    line.set_dashes([5, 2])
create_example_plots(ax=ax, base_name="campaign_stopping")
