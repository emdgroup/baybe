## Example for Modelling a Slot-Based Mixture

### Terminology

# Modelling a mixture is possible in a non-traditional way with something we refer to as
# **slots**. A slot consists of one parameter indicating the amount of a substance and
# another parameter indicating the type of substance (as label) that is in the slot.
# Contrary to traditional mixture modelling, the total number of parameters is not
# defined by how many substance choices we have, but by the maximum number of slots we
# want to allow. For instance, if we want to design a mixture with *up to five*
# components, we would need 5 slots, i.e. 10 parameters.

# A corresponding search space with three slots could look like this:
# | Slot1_Label | Slot1_Amount | Slot2_Label | Slot2_Amount | Slot3_Label | Slot3_Amount |
# |:------------|:-------------|:------------|:-------------|:------------|:-------------|
# | Solvent1    | 10           | Solvent5    | 20           | Solvent4    | 70           |
# | Solvent1    | 30           | Solvent8    | 40           | Solvent2    | 30           |
# | Solvent3    | 20           | Solvent1    | 35           | Solvent9    | 45           |
# | Solvent2    | 15           | Solvent3    | 40           | Solvent1    | 45           |

# This slot-based representation has one decided advantage compared to traditional
# modelling: We can utilize BayBE's label encodings for the label parameters. For
# instance, when mixing small molecules, the
# [`SubstanceParameter`](baybe.parameters.substance.SubstanceParameter) can be used to
# smartly encode the slot labels, enabling the algorithm to perform a chemically-aware
# mixture optimization.

# In this example, we show how to design the search space and the various discrete
# constraints we need to impose. We simulate a situation where we want to mix up to
# three solvents, i.e. we will have 3 slots (6 parameters). Their respective amounts
# need to sum up to 100. Also, a solvents should never be chosen twice, which
# requires various other constraints.

# ```{admonition} Discrete vs. Continuous Modelling
# :class: important
# In here, we utilize only discrete parameters, although in principle, the parameters
# corresponding to amounts could also be modelled as continuous numbers. This however,
# would mean some of the constraints we need act between discrete and continuous
# parameters - which is not supported at the moment.
# ```

### Imports

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
from baybe.parameters import NumericalDiscreteParameter, SubstanceParameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from baybe.utils.dataframe import add_fake_measurements

# Basic example settings.

SUM_TOLERANCE = 0.1  # The tolerance we allow for the fulfillment of sum constraints
SMOKE_TEST = "SMOKE_TEST" in os.environ
RESOLUTION = 5 if SMOKE_TEST else 11  # resolution of the discretization

### Parameter Setup

# Create parameters for the slot labels. Each of our slots offers a choice between
# 4 solvents.

dict_solvents = {
    "water": "O",
    "C1": "C",
    "C2": "CC",
    "C3": "CCC",
}
slot1_label = SubstanceParameter(
    name="Slot1_Label", data=dict_solvents, encoding="MORDRED"
)
slot2_label = SubstanceParameter(
    name="Slot2_Label", data=dict_solvents, encoding="MORDRED"
)
slot3_label = SubstanceParameter(
    name="Slot3_Label", data=dict_solvents, encoding="MORDRED"
)

# Create parameters for the slot amounts.

slot1_amount = NumericalDiscreteParameter(
    name="Slot1_Amount", values=list(np.linspace(0, 100, RESOLUTION)), tolerance=0.2
)
slot2_amount = NumericalDiscreteParameter(
    name="Slot2_Amount", values=list(np.linspace(0, 100, RESOLUTION)), tolerance=0.2
)
slot3_amount = NumericalDiscreteParameter(
    name="Slot3_Amount", values=list(np.linspace(0, 100, RESOLUTION)), tolerance=0.2
)

parameters = [
    slot1_label,
    slot2_label,
    slot3_label,
    slot1_amount,
    slot2_amount,
    slot3_amount,
]

### Constraint Setup

# Like for all mixtures, let us ensure that the overall sum of slot amounts is always
# 100.

sum_constraint = DiscreteSumConstraint(
    parameters=["Slot1_Amount", "Slot2_Amount", "Slot3_Amount"],
    condition=ThresholdCondition(threshold=100, operator="=", tolerance=SUM_TOLERANCE),
)

# We could have a situation where we do not care about the order of addition of
# components to the mixture. This comes with two additional constraints.

# If there is no order of addition, it does not matter whether we have two slots
# with the same substance or just one holding the combined amounts of two slots
# with the same slot ingredient. Thus, let us make sure that no slot contains a
# duplicate label entry.

no_duplicates_constraint = DiscreteNoLabelDuplicatesConstraint(
    parameters=["Slot1_Label", "Slot2_Label", "Slot3_Label"]
)

# Next, we need to take care of permutation invariance. If our order of addition does
# not matter, the result of exchanging slot 1 with slot 3 does not change the mixture,
# i.e. the mixture slots are permutation invariant.

# One complication arising for the permutation invariance in this case stems from the
# fact that we not only have a label per slot, but also a numerical amount. Now if
# this amount is zero, it actually does not matter what label the slot has (i.e.
# what substance should be considered for that slot), because we are adding 0 of it to
# the mixture anyway. In BayBE, we call this a "dependency", i.e. the slot labels
# depend on the slot amounts and are only relevant if the amount fulfills some
# condition (in this case "amount > 0"). The `DiscreteDependenciesConstraint` tells
# the `DiscretePermutationInvarianceConstraint` about these dependencies so that they
# are correctly included in the filtering process.

perm_inv_constraint = DiscretePermutationInvarianceConstraint(
    parameters=["Slot1_Label", "Slot2_Label", "Slot3_Label"],
    dependencies=DiscreteDependenciesConstraint(
        parameters=["Slot1_Amount", "Slot2_Amount", "Slot3_Amount"],
        conditions=[
            ThresholdCondition(threshold=0.0, operator=">"),
            ThresholdCondition(threshold=0.0, operator=">"),
            ThresholdCondition(threshold=0.0, operator=">"),
        ],
        affected_parameters=[["Slot1_Label"], ["Slot2_Label"], ["Slot3_Label"]],
    ),
)

constraints = [perm_inv_constraint, sum_constraint, no_duplicates_constraint]

# ```{admonition} Order of Addition
# :class: note
# Whether you need to impose the constraints for removing duplicates and
# permutation invariance depends on your use case. If the order of addition is relevant
# to your mixture, there is no permutation invariance and one could argue that
# duplicates should also be allowed if subsequent steps can add the same substance.
# ```

### Campaign Setup

searchspace = SearchSpace.from_product(parameters=parameters, constraints=constraints)
objective = NumericalTarget(name="Target_1", mode="MAX").to_objective()
campaign = Campaign(searchspace=searchspace, objective=objective)

### Verification of Constraints

# Now let us take a look at some recommendations for this campaign and check whether
# the constraints we imposed are indeed adhered to.

N_ITERATIONS = 2 if SMOKE_TEST else 3
for kIter in range(N_ITERATIONS):
    print(f"\n#### ITERATION {kIter+1} ####")

    print("## ASSERTS ##")
    print(
        "No. of searchspace entries where amounts do not sum to 100.0:      ",
        campaign.searchspace.discrete.exp_rep[
            ["Slot1_Amount", "Slot2_Amount", "Slot3_Amount"]
        ]
        .sum(axis=1)
        .apply(lambda x: x - 100.0)
        .abs()
        .gt(SUM_TOLERANCE)
        .sum(),
    )
    print(
        "No. of searchspace entries that have duplicate slot labels:        ",
        campaign.searchspace.discrete.exp_rep[
            ["Slot1_Label", "Slot2_Label", "Slot3_Label"]
        ]
        .nunique(axis=1)
        .ne(3)
        .sum(),
    )
    print(
        "No. of searchspace entries with permutation-invariant combinations:",
        campaign.searchspace.discrete.exp_rep[
            ["Slot1_Label", "Slot2_Label", "Slot3_Label"]
        ]
        .apply(frozenset, axis=1)
        .to_frame()
        .join(
            campaign.searchspace.discrete.exp_rep[
                ["Slot1_Amount", "Slot2_Amount", "Slot3_Amount"]
            ]
        )
        .duplicated()
        .sum(),
    )
    # The following asserts only work if the tolerance for the threshold condition in
    # the constraint are not 0. Otherwise, the sum/prod constraints will remove more
    # points than intended due to numeric rounding
    print(
        f"No. of unique 1-solvent entries (exp. {math.comb(len(dict_solvents), 1)*1})",
        (
            campaign.searchspace.discrete.exp_rep[
                ["Slot1_Amount", "Slot2_Amount", "Slot3_Amount"]
            ]
            == 0.0
        )
        .sum(axis=1)
        .eq(2)
        .sum(),
    )
    print(
        f"No. of unique 2-solvent entries (exp."
        f" {math.comb(len(dict_solvents), 2)*(RESOLUTION-2)})",
        (
            campaign.searchspace.discrete.exp_rep[
                ["Slot1_Amount", "Slot2_Amount", "Slot3_Amount"]
            ]
            == 0.0
        )
        .sum(axis=1)
        .eq(1)
        .sum(),
    )
    print(
        f"No. of unique 3-solvent entries (exp."
        f" {math.comb(len(dict_solvents), 3)*((RESOLUTION-3)*(RESOLUTION-2))//2})",
        (
            campaign.searchspace.discrete.exp_rep[
                ["Slot1_Amount", "Slot2_Amount", "Slot3_Amount"]
            ]
            == 0.0
        )
        .sum(axis=1)
        .eq(0)
        .sum(),
    )

    rec = campaign.recommend(batch_size=5)
    add_fake_measurements(rec, campaign.targets)
    campaign.add_measurements(rec)
