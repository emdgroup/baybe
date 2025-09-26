# # Chemical Reaction Optimization with Solvent Constraints

# In this example, we demonstrate the use of **interpoint constraints** in a chemical
# optimization scenario. We optimize reaction conditions for a batch of chemical
# experiments where exactly 60 mL of solvent must be used across the entire batch,
# while not using more than 30 mol% of catalyst across the batch.

# This scenario illustrates a common challenge in laboratory settings.
# First, it demonstrates how to enforce a **solvent requirement**:
# Exactly 60 mL of the solvent must be used across the entire batch
# since the solvent is supplied in a fixed volume container and cannot be used later.
# Second, it shows how to also include a **Catalyst loading** constraint for balancing
# the catalyst loading across experiments for cost efficiency.

# This example demonstrates how to use interpoint constraints and intrapoint constraints.
# An intrapoint constraint, often simply referred to as a constraint, applies to each individual
# experiment, ensuring that certain conditions are met within that single point.
# In contrast, an interpoint constraint applies across a batch of experiments,
# enforcing conditions that relate to the collective set of points rather than
# individual ones. These constraints are particularly useful when resources or conditions must be
# managed at a batch level and they allow us to:
# * Ensure total resource consumption meets exact requirements
# * Maintain chemical balances across multiple experiments
# * Optimize the collective use of expensive materials

# For more details on interpoint constraints, see the {ref}`user guide on constraints
# <userguide/constraints:ContinuousLinearConstraint>`.

# ## Imports and Settings

import os

import pandas as pd
from matplotlib import pyplot as plt

from baybe import Campaign
from baybe.constraints import ContinuousLinearConstraint
from baybe.parameters import NumericalContinuousParameter
from baybe.recommenders import BotorchRecommender
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from baybe.utils.dataframe import add_fake_measurements
from baybe.utils.random import set_random_seed

SMOKE_TEST = "SMOKE_TEST" in os.environ
BATCH_SIZE = 3
N_ITERATIONS = 4 if SMOKE_TEST else 15
TOLERANCE = 0.01


set_random_seed(42)

# ## Defining the Chemical Optimization Problem

# We'll optimize a synthetic chemical reaction with the following experimental parameters:
# - **Solvent Volume** (10-30 mL per experiment): The amount of solvent used
# - **Reactant A Concentration** (0.1-2.0 g/L): Primary reactant concentration
# - **Catalyst Loading** (1-10 mol%): Catalyst amount as percentage of limiting reagent
# - **Temperature** (60-120 °C): Reaction temperature
# Note that these ranges are chosen arbitrary and do not represent a specific real-world reaction.

parameters = [
    NumericalContinuousParameter(
        name="Solvent_Volume", bounds=(10.0, 30.0), metadata={"unit": "mL"}
    ),
    NumericalContinuousParameter(
        name="Reactant_A_Conc", bounds=(0.1, 2.0), metadata={"unit": "g/L"}
    ),
    NumericalContinuousParameter(
        name="Catalyst_Loading", bounds=(1.0, 10.0), metadata={"unit": "mol%"}
    ),
    NumericalContinuousParameter(
        name="Temperature", bounds=(60.0, 120.0), metadata={"unit": "°C"}
    ),
]

# ## Constraint Definition

# We define both intrapoint and interpoint constraints to demonstrate the difference:
#
# **Intrapoint constraint** (applied to each individual experiment):
# - Reagent efficiency: For each experiment, solvent volume must be at least 5 times
#   the reactant concentration (to ensure proper dilution)
#
# **Interpoint constraints** (applied across the entire batch):
# 1. **Solvent constraint**: Total solvent across all experiments must equal exactly 60 mL
# 2. **Catalyst budget**: Total catalyst loading across batch should be ≤ 30 mol%

intrapoint_constraints = [
    ContinuousLinearConstraint(
        parameters=["Solvent_Volume", "Reactant_A_Conc"],
        operator=">=",
        coefficients=[1, -5],
        rhs=0.0,
        interpoint=False,
    ),
]

interpoint_constraints = [
    ContinuousLinearConstraint(
        parameters=["Solvent_Volume"],
        operator="=",
        coefficients=[1],
        rhs=60.0,
        interpoint=True,
    ),
    ContinuousLinearConstraint(
        parameters=["Catalyst_Loading"],
        operator="<=",
        coefficients=[1],
        rhs=30.0,
        interpoint=True,
    ),
]

# ## Campaign Setup

# We construct the search space by combining parameters with constraints, then create
# a campaign targeting maximum reaction yield. The
# {class}`~baybe.recommenders.pure.bayesian.botorch.BotorchRecommender` with
# `sequential_continuous=False` is required for interpoint constraints as they
# operate on batches rather than individual experiments.

searchspace = SearchSpace.from_product(
    parameters=parameters,
    constraints=intrapoint_constraints + interpoint_constraints,
)

target = NumericalTarget(name="Reaction_Yield")
objective = target.to_objective()

# ## Measurement Simulation

# For this example, we use the `add_fake_measurements` utility to generate
# synthetic target values. This utility creates random measurements within
# the target's expected range, which is useful for testing and demonstration
# purposes without requiring a complex reaction model.

recommender = BotorchRecommender(sequential_continuous=False)

campaign = Campaign(
    searchspace=searchspace,
    objective=objective,
    recommender=recommender,
)

# ## Initial Training Data

# We generate 5 random experiments from the search space to simulate existing data.

initial_data = searchspace.continuous.sample_uniform(5)
add_fake_measurements(initial_data, campaign.targets)
campaign.add_measurements(initial_data)

# ## Optimization Loop with Constraint Validation

# We run several optimization iterations, where each iteration recommends a batch
# of experiments that satisfy both intrapoint and interpoint constraints. After
# evaluating each batch, we validate that the interpoint constraints are satisfied
# and use assertions to ensure the optimization respects our resource limitations.

results_log = []

for it in range(N_ITERATIONS):
    recommendations = campaign.recommend(batch_size=BATCH_SIZE)

    add_fake_measurements(recommendations, campaign.targets)
    campaign.add_measurements(recommendations)
    total_sol = recommendations["Solvent_Volume"].sum()
    total_cat = recommendations["Catalyst_Loading"].sum()
    solvent_ok = abs(total_sol - 60.0) < TOLERANCE
    catalyst_ok = total_cat <= (30.0 + TOLERANCE)

    assert solvent_ok, (
        f"Solvent constraint violated: {total_sol:.1f} mL (expected 60.0 mL)"
    )
    assert catalyst_ok, (
        f"Catalyst constraint violated: {total_cat:.1f} mol% (max 30.0 mol%)"
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

# We create plots showing both individual experiment values and their totals to
# illustrate how interpoint constraints work. The individual lines show how the
# optimizer distributes resources across experiments within each batch, while the
# bold total lines demonstrate that the batch-level constraints are satisfied.

results_df = pd.DataFrame(results_log)

# fmt: off
fig, axs = plt.subplots(1, 2, figsize=(10, 4));
# fmt: on

plt.sca(axs[0])
for exp_idx in range(BATCH_SIZE):
    individual_values = [
        batch[exp_idx] for batch in results_df["individual_solvent_mL"]
    ]
    plt.plot(
        results_df["iteration"],
        individual_values,
        "o-",
        alpha=0.6,
        label=f"Exp {exp_idx + 1}",
    )

plt.plot(
    results_df["iteration"],
    results_df["total_solvent_mL"],
    "s-",
    color="blue",
    linewidth=2,
    label="Total",
)
plt.axhline(y=60, color="red", linestyle="--", label="Required")
plt.title("Solvent: Individual + Total")
# fmt: off
plt.legend();
# fmt: on

plt.sca(axs[1])
for exp_idx in range(BATCH_SIZE):
    individual_values = [
        batch[exp_idx] for batch in results_df["individual_catalyst_mol%"]
    ]
    plt.plot(
        results_df["iteration"],
        individual_values,
        "o-",
        alpha=0.6,
        label=f"Exp {exp_idx + 1}",
    )

plt.plot(
    results_df["iteration"],
    results_df["total_catalyst_mol%"],
    "s-",
    color="orange",
    linewidth=2,
    label="Total",
)
plt.axhline(y=30, color="red", linestyle="--", label="Limit")
plt.title("Catalyst: Individual + Total")
# fmt: off
plt.legend();
# fmt: on

plt.tight_layout()
plt.show()
