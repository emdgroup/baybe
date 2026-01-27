# # Chemical Reaction Optimization with Catalyst Constraints

# This example demonstrates the use of **interpoint constraints**, which can be used to
# apply restrictions on batches of experiments rather than on individual experiments.
# For more details on interpoint constraints, we refer to the {ref}`constraints user
# guide <userguide/constraints:Interpoint Constraints>`.

# ## The Scenario

# For the example, we consider a typical chemical optimization scenario. The goal is to
# optimize reaction conditions for a batch of chemical experiments where exactly 30 mol%
# of a certain catalyst must be used for the entire batch while not using more than
# 60 mL of solvent in total.

# This scenario illustrates two common challenges in laboratory settings:
# 1. First, it demonstrates how to enforce a **catalyst requirement**: Exactly 30 mol%
# of the catalyst must be used across the entire batch since the catalyst is supplied in
# a sealed sensitive package that cannot be reused once opened.
# 2. Second, it shows how to include a **solvent budget** constraint for
# controlling the total solvent consumption across experiments for cost efficiency.

# ## Imports and Settings

import os

import pandas as pd
from matplotlib import pyplot as plt

from baybe import Campaign
from baybe.constraints import ContinuousLinearConstraint
from baybe.parameters import NumericalContinuousParameter
from baybe.recommenders import BotorchRecommender
from baybe.recommenders.meta.sequential import TwoPhaseMetaRecommender
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from baybe.utils.dataframe import add_fake_measurements
from baybe.utils.random import set_random_seed

SMOKE_TEST = "SMOKE_TEST" in os.environ
BATCH_SIZE = 3
N_ITERATIONS = 2 if SMOKE_TEST else 15
TOLERANCE = 0.01


set_random_seed(1337)

# ## Problem Definition

# We consider a synthetic chemical reaction with the following experimental parameters:
# - **Solvent Volume** (10-30 mL per experiment): The amount of solvent used
# - **Reactant Concentration** (0.1-2.0 g/L): Primary reactant concentration
# - **Catalyst Loading** (1-20 mol%): Catalyst amount as percentage of limiting reagent
# - **Temperature** (60-120 °C): Reaction temperature
#
# Note that the parameter ranges are chosen arbitrarily and do not correspond to any
# specific real-world reaction.

parameters = [
    NumericalContinuousParameter(
        name="Solvent_Volume", bounds=(10.0, 30.0), metadata={"unit": "mL"}
    ),
    NumericalContinuousParameter(
        name="Reactant_Conc", bounds=(0.1, 2.0), metadata={"unit": "g/L"}
    ),
    NumericalContinuousParameter(
        name="Catalyst_Loading", bounds=(1.0, 20.0), metadata={"unit": "mol%"}
    ),
    NumericalContinuousParameter(
        name="Temperature", bounds=(60.0, 120.0), metadata={"unit": "°C"}
    ),
]

# ## Constraint Definition

# For the example, the following constraints are applied:
#
# 1. **Reagent efficiency**: For each experiment, the solvent volume must be at least
#    five times the reactant concentration (to ensure proper dilution).
# 2. **Catalyst constraint**: The total catalyst loading across all experiments in a
#    batch must equal exactly 30 mol%.
# 3. **Solvent budget**: The total solvent used for each batch should be at most
#    60 mL.

# The first constraint is an *intrapoint* constraint since it applies to individual
# experiments. The latter two are *interpoint* constraints as they apply to a batch as a
# whole.

intrapoint_constraints = [
    ContinuousLinearConstraint(
        parameters=["Solvent_Volume", "Reactant_Conc"],
        operator=">=",
        coefficients=[1, -5],
        rhs=0.0,
    ),
]

interpoint_constraints = [
    ContinuousLinearConstraint(
        parameters=["Catalyst_Loading"],
        operator="=",
        coefficients=[1],
        rhs=30.0,
        interpoint=True,
    ),
    ContinuousLinearConstraint(
        parameters=["Solvent_Volume"],
        operator="<=",
        coefficients=[1],
        rhs=60.0,
        interpoint=True,
    ),
]

# ## Campaign Setup

# With these components in place, we can now define our search space and set up the
# corresponding experimental campaign:

searchspace = SearchSpace.from_product(
    parameters=parameters,
    constraints=intrapoint_constraints + interpoint_constraints,
)
objective = NumericalTarget(name="Reaction_Yield").to_objective()
recommender = TwoPhaseMetaRecommender(
    recommender=BotorchRecommender(sequential_continuous=False)
)
campaign = Campaign(
    searchspace=searchspace,
    objective=objective,
    recommender=recommender,
)


# ## Optimization Loop with Constraint Validation

# Next, we run several optimization iterations and validate that the interpoint
# constraints are satisfied for each batch to ensure the optimization respects
# our resource limitations:

results_log = []

for it in range(N_ITERATIONS):
    recommendations = campaign.recommend(batch_size=BATCH_SIZE)

    add_fake_measurements(recommendations, campaign.targets)
    campaign.add_measurements(recommendations)
    total_sol = recommendations["Solvent_Volume"].sum()
    total_cat = recommendations["Catalyst_Loading"].sum()
    solvent_ok = total_sol <= (60.0 + TOLERANCE)
    catalyst_ok = abs(total_cat - 30.0) < TOLERANCE

    assert solvent_ok, f"Solvent constraint violated: {total_sol:.1f} mL (max 60.0 mL)"
    assert catalyst_ok, (
        f"Catalyst constraint violated: {total_cat:.1f} mol% (expected 30.0 mol%)"
    )

    results_log.append(
        {
            "iteration": it + 1,
            "total_solvent_mL": total_sol,
            "total_catalyst_mol%": total_cat,
            "individual_solvent_mL": recommendations["Solvent_Volume"].tolist(),
            "individual_catalyst_mol%": recommendations["Catalyst_Loading"].tolist(),
        }
    )

# ## Visualization

# The plots below show the parameter values of individual experiments (labeled `Exp <n>`)
# and as well as their totals over the respective batch (labeled `Total`), to
# illustrate the effects of the two interpoint constraints:
# * The total solvent used for any individual batch never exceeds the given budget of
# 60 mL, even though the exact amount varies across batches and distributes differently
# among individual experiments.
# * By constrast, the total catalyst loading per batch is always
# exactly 30 mol%, even though individual experiments have varying amounts.

# fmt: off
fig, axs = plt.subplots(1, 2, figsize=(10, 4));
# fmt: on

plot_configs = [
    {
        "ax": axs[0],
        "individual_col": "individual_solvent_mL",
        "total_col": "total_solvent_mL",
        "y": 60,
        "label": "Budget",
        "title": "Solvent",
        "ylabel": "Solvent Volume (mL)",
    },
    {
        "ax": axs[1],
        "individual_col": "individual_catalyst_mol%",
        "total_col": "total_catalyst_mol%",
        "y": 30,
        "label": "Required",
        "title": "Catalyst",
        "ylabel": "Catalyst Loading (mol%)",
    },
]

results_df = pd.DataFrame(results_log)
for config in plot_configs:
    plt.sca(config["ax"])

    for exp_idx, values_per_exp in enumerate(
        zip(*results_df[config["individual_col"]]), start=1
    ):
        plt.plot(
            results_df["iteration"],
            values_per_exp,
            "o-",
            label=f"Exp {exp_idx}",
        )

    plt.plot(
        results_df["iteration"],
        results_df[config["total_col"]],
        "s-",
        label="Total",
        zorder=0,
    )

    plt.axhline(y=config["y"], label=config["label"], color="black")
    plt.ylim(bottom=0)
    plt.title(config["title"])
    plt.xlabel("Batch")
    plt.ylabel(config["ylabel"])
    # fmt: off
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));
    # fmt: on
    plt.tight_layout()

if not SMOKE_TEST:
    plt.savefig("interpoint.svg")
