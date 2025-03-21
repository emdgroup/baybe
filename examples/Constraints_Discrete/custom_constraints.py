## Example for using custom constraints in discrete searchspaces

# This examples shows how a custom constraint can be created for a discrete searchspace.
# That is, it shows how the user can define a constraint restricting the searchspace.

# This example assumes some basic familiarity with using BayBE.
# We thus refer to [`campaign`](./../Basics/campaign.md) for a basic example.

### Necessary imports for this example

import os

import numpy as np
import pandas as pd

from baybe import Campaign
from baybe.constraints import DiscreteCustomConstraint
from baybe.objectives import SingleTargetObjective
from baybe.parameters import (
    CategoricalParameter,
    NumericalDiscreteParameter,
    SubstanceParameter,
)
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from baybe.utils.dataframe import add_fake_measurements

### Experiment setup

# We begin by setting up some parameters for our experiments.
# `TEMPERATURE_RESOLUTION` describes the number of different temperatures used.

SMOKE_TEST = "SMOKE_TEST" in os.environ
TEMPERATURE_RESOLUTION = 3 if SMOKE_TEST else 10

dict_solvent = {
    "water": "O",
    "C1": "C",
    "C2": "CC",
    "C3": "CCC",
    "C4": "CCCC",
    "C5": "CCCCC",
    "c6": "c1ccccc1",
    "C6": "CCCCCC",
}
solvent = SubstanceParameter("Solvent", data=dict_solvent, encoding="RDKIT")
speed = CategoricalParameter(
    "Speed", values=["very slow", "slow", "normal", "fast", "very fast"], encoding="INT"
)
temperature = NumericalDiscreteParameter(
    "Temperature",
    values=list(np.linspace(100, 200, TEMPERATURE_RESOLUTION)),
    tolerance=0.5,
)
concentration = NumericalDiscreteParameter(
    "Concentration", values=[1, 2, 5, 10], tolerance=0.4
)

parameters = [solvent, speed, temperature, concentration]

### Creating the constraint

# The constraints are handled when creating the searchspace object.
# We thus need to define our constraint first as follows.


def custom_function(df: pd.DataFrame) -> pd.Series:
    """This constraint implements a custom user-defined filter/validation functionality."""  # noqa: D401
    # Situation 1: We only want entries where the solvent water is used with
    # temperatures <= 120 and concentrations <= 5
    mask_bad1 = (
        (df["Solvent"] == "water")
        & (df["Temperature"] > 120)
        & (df["Concentration"] > 5)
    )

    # Situation 2: We only want entries where the solvent C2 is used with
    # temperatures <= 180 and concentrations <= 3
    mask_bad2 = (
        (df["Solvent"] == "C2") & (df["Temperature"] > 180) & (df["Concentration"] > 3)
    )

    # Situation 3: We only want entries where the solvent C3 is used with
    # temperatures <= 150 and concentrations <= 3
    mask_bad3 = (
        (df["Solvent"] == "C3") & (df["Temperature"] > 150) & (df["Concentration"] > 3)
    )

    # Combine all situations
    mask_good = ~(mask_bad1 | mask_bad2 | mask_bad3)

    return mask_good


# We now initialize the `CustomConstraint` with all parameters this function should have access to.

constraint = DiscreteCustomConstraint(
    parameters=["Concentration", "Solvent", "Temperature"], validator=custom_function
)

### Creating the searchspace and the objective

searchspace = SearchSpace.from_product(parameters=parameters, constraints=[constraint])

objective = SingleTargetObjective(target=NumericalTarget(name="yield", mode="MAX"))

### Creating and printing the campaign

campaign = Campaign(searchspace=searchspace, objective=objective)
print(campaign)

### Manual verification of the constraint

# The following loop performs some recommendations and manually verifies the given constraints.

N_ITERATIONS = 3
for kIter in range(N_ITERATIONS):
    print(f"\n\n#### ITERATION {kIter+1} ####")

    print("## ASSERTS ##")
    print(
        "Number of entries with water, temp > 120 and concentration > 5:      ",
        (
            campaign.searchspace.discrete.exp_rep["Concentration"].apply(
                lambda x: x > 5
            )
            & campaign.searchspace.discrete.exp_rep["Temperature"].apply(
                lambda x: x > 120
            )
            & campaign.searchspace.discrete.exp_rep["Solvent"].eq("water")
        ).sum(),
    )
    print(
        "Number of entries with C2, temp > 180 and concentration > 3:         ",
        (
            campaign.searchspace.discrete.exp_rep["Concentration"].apply(
                lambda x: x > 3
            )
            & campaign.searchspace.discrete.exp_rep["Temperature"].apply(
                lambda x: x > 180
            )
            & campaign.searchspace.discrete.exp_rep["Solvent"].eq("C2")
        ).sum(),
    )
    print(
        "Number of entries with C3, temp > 150 and concentration > 3:         ",
        (
            campaign.searchspace.discrete.exp_rep["Concentration"].apply(
                lambda x: x > 3
            )
            & campaign.searchspace.discrete.exp_rep["Temperature"].apply(
                lambda x: x > 150
            )
            & campaign.searchspace.discrete.exp_rep["Solvent"].eq("C3")
        ).sum(),
    )

    rec = campaign.recommend(batch_size=5)
    add_fake_measurements(rec, campaign.targets)
    campaign.add_measurements(rec)
