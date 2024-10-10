## Example for using a mixture use case in a discrete searchspace

# Example for imposing sum constraints for discrete parameters.
# The constraints simulate a situation where we want to mix up to three solvents.
# However, their respective fractions need to sum up to 100.
# Also, the solvents should never be chosen twice, which requires various other constraints.

# This example assumes some basic familiarity with using BayBE.
# We thus refer to [`campaign`](./../Basics/campaign.md) for a basic example.

### Necessary imports for this example

import math
import os

import numpy as np

from baybe import Campaign
from baybe.constraints import (
    DiscreteDependenciesConstraint,
    DiscreteNoLabelDuplicatesConstraint,
    DiscretePermutationInvarianceConstraint,
    DiscreteSumConstraint,
    ThresholdCondition,
)
from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalDiscreteParameter, SubstanceParameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from baybe.utils.dataframe import add_fake_measurements

### Experiment setup

# This parameter denotes the tolerance with regard to the calculation of the sum.

SUM_TOLERANCE = 1.0

SMOKE_TEST = "SMOKE_TEST" in os.environ

# This parameter denotes the resolution of the discretization of the parameters
RESOLUTION = 5 if SMOKE_TEST else 12

dict_solvents = {
    "water": "O",
    "C1": "C",
    "C2": "CC",
    "C3": "CCC",
}
solvent1 = SubstanceParameter(name="Solv1", data=dict_solvents, encoding="MORDRED")
solvent2 = SubstanceParameter(name="Solv2", data=dict_solvents, encoding="MORDRED")
solvent3 = SubstanceParameter(name="Solv3", data=dict_solvents, encoding="MORDRED")

# Parameters for representing the fraction.

fraction1 = NumericalDiscreteParameter(
    name="Frac1", values=list(np.linspace(0, 100, RESOLUTION)), tolerance=0.2
)
fraction2 = NumericalDiscreteParameter(
    name="Frac2", values=list(np.linspace(0, 100, RESOLUTION)), tolerance=0.2
)
fraction3 = NumericalDiscreteParameter(
    name="Frac3", values=list(np.linspace(0, 100, RESOLUTION)), tolerance=0.2
)

parameters = [solvent1, solvent2, solvent3, fraction1, fraction2, fraction3]

### Creating the constraint

# Since the constraints are required for the creation of the searchspace, we create
# them next.
# Note that we need a `PermutationInvarianceConstraint` here.
# The reason is that constraints are normally applied in a specific order.
# However, the fractions should be invariant under permutations.
# We thus require an explicit constraint for this.

perm_inv_constraint = DiscretePermutationInvarianceConstraint(
    parameters=["Solv1", "Solv2", "Solv3"],
    dependencies=DiscreteDependenciesConstraint(
        parameters=["Frac1", "Frac2", "Frac3"],
        conditions=[
            ThresholdCondition(threshold=0.0, operator=">"),
            ThresholdCondition(threshold=0.0, operator=">"),
            ThresholdCondition(threshold=0.0, operator=">"),
        ],
        affected_parameters=[["Solv1"], ["Solv2"], ["Solv3"]],
    ),
)

# This is now the actual sum constraint

sum_constraint = DiscreteSumConstraint(
    parameters=["Frac1", "Frac2", "Frac3"],
    condition=ThresholdCondition(threshold=100, operator="=", tolerance=SUM_TOLERANCE),
)

# The permutation invariance might create duplicate labels.
# We thus include a constraint to remove them.

no_duplicates_constraint = DiscreteNoLabelDuplicatesConstraint(
    parameters=["Solv1", "Solv2", "Solv3"]
)

constraints = [perm_inv_constraint, sum_constraint, no_duplicates_constraint]

### Creating the searchspace and the objective

searchspace = SearchSpace.from_product(parameters=parameters, constraints=constraints)

objective = SingleTargetObjective(target=NumericalTarget(name="Target_1", mode="MAX"))

### Creating and printing the campaign

campaign = Campaign(searchspace=searchspace, objective=objective)
print(campaign)

### Manual verification of the constraint

# The following loop performs some recommendations and manually verifies the given constraints.

N_ITERATIONS = 2 if SMOKE_TEST else 3
for kIter in range(N_ITERATIONS):
    print(f"\n#### ITERATION {kIter+1} ####")

    print("## ASSERTS ##")
    print(
        "No. of searchspace entries where fractions do not sum to 100.0:      ",
        campaign.searchspace.discrete.exp_rep[["Frac1", "Frac2", "Frac3"]]
        .sum(axis=1)
        .apply(lambda x: x - 100.0)
        .abs()
        .gt(SUM_TOLERANCE)
        .sum(),
    )
    print(
        "No. of searchspace entries that have duplicate solvent labels:       ",
        campaign.searchspace.discrete.exp_rep[["Solv1", "Solv2", "Solv3"]]
        .nunique(axis=1)
        .ne(3)
        .sum(),
    )
    print(
        "No. of searchspace entries with permutation-invariant combinations:  ",
        campaign.searchspace.discrete.exp_rep[["Solv1", "Solv2", "Solv3"]]
        .apply(frozenset, axis=1)
        .to_frame()
        .join(campaign.searchspace.discrete.exp_rep[["Frac1", "Frac2", "Frac3"]])
        .duplicated()
        .sum(),
    )
    # The following asserts only work if the tolerance for the threshold condition in
    # the constraint are not 0. Otherwise, the sum/prod constraints will remove more
    # points than intended due to numeric rounding
    print(
        f"No. of unique 1-solvent entries (exp. {math.comb(len(dict_solvents), 1)*1})",
        (campaign.searchspace.discrete.exp_rep[["Frac1", "Frac2", "Frac3"]] == 0.0)
        .sum(axis=1)
        .eq(2)
        .sum(),
    )
    print(
        f"No. of unique 2-solvent entries (exp."
        f" {math.comb(len(dict_solvents), 2)*(RESOLUTION-2)})",
        (campaign.searchspace.discrete.exp_rep[["Frac1", "Frac2", "Frac3"]] == 0.0)
        .sum(axis=1)
        .eq(1)
        .sum(),
    )
    print(
        f"No. of unique 3-solvent entries (exp."
        f" {math.comb(len(dict_solvents), 3)*((RESOLUTION-3)*(RESOLUTION-2))//2})",
        (campaign.searchspace.discrete.exp_rep[["Frac1", "Frac2", "Frac3"]] == 0.0)
        .sum(axis=1)
        .eq(0)
        .sum(),
    )

    rec = campaign.recommend(batch_size=5)
    add_fake_measurements(rec, campaign.targets)
    campaign.add_measurements(rec)
