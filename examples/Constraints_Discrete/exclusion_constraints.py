## Example for using exclusion constraints in discrete searchspaces

# This examples shows how an exclusion constraint can be created for a discrete searchspace.
# This can be used if some parameter values are incompatible with values of another parameter.

# This example assumes some basic familiarity with using BayBE.
# We thus refer to [`campaign`](./../Basics/campaign.md) for a basic example.

### Necessary imports for this example

import numpy as np

from baybe import Campaign
from baybe.constraints import (
    DiscreteExcludeConstraint,
    SubSelectionCondition,
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

# We begin by setting up some parameters for our experiments.

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
solvent = SubstanceParameter(name="Solv", data=dict_solvent, encoding="RDKIT")
speed = CategoricalParameter(
    name="Speed",
    values=["very slow", "slow", "normal", "fast", "very fast"],
    encoding="INT",
)
temperature = NumericalDiscreteParameter(
    name="Temp", values=list(np.linspace(100, 200, 15)), tolerance=0.4
)
pressure = NumericalDiscreteParameter(
    name="Pressure", values=[1, 2, 5, 10], tolerance=0.4
)

parameters = [solvent, speed, temperature, pressure]

### Creating the constraint

# This constraint simulates a situation where solvents `C2` and `C4` are not
# compatible with temperatures larger than 151 and should thus be excluded.

constraint_1 = DiscreteExcludeConstraint(
    parameters=["Temp", "Solv"],
    combiner="AND",
    conditions=[
        ThresholdCondition(threshold=151, operator=">"),
        SubSelectionCondition(selection=["C4", "C2"]),
    ],
)

# This constraint simulates a situation where solvents `C5` and `C6` are not
# compatible with pressures larger than 5 and should thus be excluded.

constraint_2 = DiscreteExcludeConstraint(
    parameters=["Pressure", "Solv"],
    combiner="AND",
    conditions=[
        ThresholdCondition(threshold=5, operator=">"),
        SubSelectionCondition(selection=["C5", "C6"]),
    ],
)

# This constraint simulates a situation where pressures below 3 should never be
# combined with temperatures above 120.

constraint_3 = DiscreteExcludeConstraint(
    parameters=["Pressure", "Temp"],
    combiner="AND",
    conditions=[
        ThresholdCondition(threshold=3.0, operator="<"),
        ThresholdCondition(threshold=120.0, operator=">"),
    ],
)

### Creating the searchspace and the objective

# We now create the searchspace using the previously defined constraints.

searchspace = SearchSpace.from_product(
    parameters=parameters, constraints=[constraint_1, constraint_2, constraint_3]
)

objective = SingleTargetObjective(target=NumericalTarget(name="Target_1", mode="MAX"))

### Creating and printing the campaign
campaign = Campaign(searchspace=searchspace, objective=objective)
print(campaign)


### Manual verification of the constraints

# The following loop performs some iterations and manually verifies the given constraints.

N_ITERATIONS = 3
for kIter in range(N_ITERATIONS):
    print(f"\n\n#### ITERATION {kIter+1} ####")

    print("## ASSERTS ##")
    print(
        "Number of entries with either Solvents C2 or C4 and a temperature above 151: ",
        (
            campaign.searchspace.discrete.exp_rep["Temp"].apply(lambda x: x > 151)
            & campaign.searchspace.discrete.exp_rep["Solv"].apply(
                lambda x: x in ["C2", "C4"]
            )
        ).sum(),
    )
    print(
        "Number of entries with either Solvents C5 or C6 and a pressure above 5:      ",
        (
            campaign.searchspace.discrete.exp_rep["Pressure"].apply(lambda x: x > 5)
            & campaign.searchspace.discrete.exp_rep["Solv"].apply(
                lambda x: x in ["C5", "C6"]
            )
        ).sum(),
    )
    print(
        "Number of entries with pressure below 3 and temperature above 120:           ",
        (
            campaign.searchspace.discrete.exp_rep["Pressure"].apply(lambda x: x < 3)
            & campaign.searchspace.discrete.exp_rep["Temp"].apply(lambda x: x > 120)
        ).sum(),
    )

    rec = campaign.recommend(batch_size=5)
    add_fake_measurements(rec, campaign.targets)
    campaign.add_measurements(rec)
