## Example for using exclusion constraints incorporating sums and products

# This examples demonstrates an exclusion constraint using products and sums.

# This example assumes some basic familiarity with using BayBE.
# We thus refer to [`campaign`](./../Basics/campaign.md) for a basic example.

### Necessary imports for this example

import os

import numpy as np

from baybe import Campaign
from baybe.constraints import (
    DiscreteProductConstraint,
    DiscreteSumConstraint,
    ThresholdCondition,
)
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

SMOKE_TEST = "SMOKE_TEST" in os.environ

RESOLUTION = 3 if SMOKE_TEST else 5

dict_solvent = {
    "water": "O",
    "C1": "C",
    "C2": "CC",
}
solvent = SubstanceParameter(name="Solvent", data=dict_solvent, encoding="RDKIT")
speed = CategoricalParameter(
    name="Speed", values=["slow", "normal", "fast"], encoding="INT"
)
num_parameter_1 = NumericalDiscreteParameter(
    name="NumParam1", values=list(np.linspace(0, 100, RESOLUTION)), tolerance=0.5
)
num_parameter_2 = NumericalDiscreteParameter(
    name="NumParam2", values=list(np.linspace(0, 100, RESOLUTION)), tolerance=0.5
)
num_parameter_3 = NumericalDiscreteParameter(
    name="NumParam3", values=list(np.linspace(0, 100, RESOLUTION)), tolerance=0.5
)
num_parameter_4 = NumericalDiscreteParameter(
    name="NumParam4", values=list(np.linspace(0, 100, RESOLUTION)), tolerance=0.5
)
num_parameter_5 = NumericalDiscreteParameter(
    name="NumParam5", values=list(np.linspace(0, 100, RESOLUTION)), tolerance=0.5
)
num_parameter_6 = NumericalDiscreteParameter(
    name="NumParam6", values=list(np.linspace(0, 100, RESOLUTION)), tolerance=0.5
)

parameters = [
    solvent,
    speed,
    num_parameter_1,
    num_parameter_2,
    num_parameter_3,
    num_parameter_4,
    num_parameter_5,
    num_parameter_6,
]

### Creating the constraints

# Constraints are used when creating the searchspace object.
# Thus, they need to be defined prior to the searchspace creation.

sum_constraint_1 = DiscreteSumConstraint(
    parameters=["NumParam1", "NumParam2"],
    condition=ThresholdCondition(threshold=150.0, operator="<="),
)
sum_constraint_2 = DiscreteSumConstraint(
    parameters=["NumParam5", "NumParam6"],
    condition=ThresholdCondition(threshold=100, operator="=", tolerance=1.0),
)
prod_constraint = DiscreteProductConstraint(
    parameters=["NumParam3", "NumParam4"],
    condition=ThresholdCondition(threshold=30, operator=">="),
)

constraints = [sum_constraint_1, sum_constraint_2, prod_constraint]

### Creating the searchspace and the objective

searchspace = SearchSpace.from_product(parameters=parameters, constraints=constraints)

objective = SingleTargetObjective(target=NumericalTarget(name="Target_1", mode="MAX"))

### Creating and printing the campaign

campaign = Campaign(searchspace=searchspace, objective=objective)
print(campaign)

### Manual verification of the constraints

# The following loop performs some recommendations and manually verifies the given constraints.

N_ITERATIONS = 2 if SMOKE_TEST else 5
for kIter in range(N_ITERATIONS):
    print(f"\n\n#### ITERATION {kIter+1} ####")

    print("## ASSERTS ##")
    print(
        "Number of entries with 1,2-sum above 150:      ",
        (
            campaign.searchspace.discrete.exp_rep[["NumParam1", "NumParam2"]].sum(
                axis=1
            )
            > 150.0
        ).sum(),
    )
    print(
        "Number of entries with 3,4-product under 30:   ",
        (
            campaign.searchspace.discrete.exp_rep[["NumParam3", "NumParam4"]].prod(
                axis=1
            )
            < 30
        ).sum(),
    )
    print(
        "Number of entries with 5,6-sum unequal to 100: ",
        campaign.searchspace.discrete.exp_rep[["NumParam5", "NumParam6"]]
        .sum(axis=1)
        .apply(lambda x: x - 100.0)
        .abs()
        .gt(0.01)
        .sum(),
    )

    rec = campaign.recommend(batch_size=5)
    add_fake_measurements(rec, campaign.targets)
    campaign.add_measurements(rec)
