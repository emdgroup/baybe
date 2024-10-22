## Modeling a Slot-Based Mixture

### Terminology

# Modeling a mixture is possible in a non-traditional way by using a concept we
# refer to as a **slot**. A slot is represented through the combination of two
# parameters: one indicating the amount of a mixture ingredient, and another indicating
# the type of ingredient (as a label) populating the slot. Unlike [traditional
# mixture modeling](/examples/Mixtures/traditional.md), the total number of parameters
# is not determined by how many ingredient choices we have, but by the maximum number of
# slots we allow. For instance, if we want to design a mixture with *up to three*
# ingredients, we can do so by creating three slots represented by six parameters.

# A corresponding search space could look like this:
# | Slot1_Label | Slot1_Amount | Slot2_Label | Slot2_Amount | Slot3_Label | Slot3_Amount |
# |:------------|:-------------|:------------|:-------------|:------------|:-------------|
# | Solvent1    | 10           | Solvent5    | 20           | Solvent4    | 30           |
# | Solvent1    | 30           | Solvent8    | 40           | Solvent2    | 30           |
# | Solvent3    | 20           | Solvent1    | 35           | Solvent9    | 30           |
# | Solvent2    | 15           | Solvent3    | 10           | Solvent1    | 30           |

# The slot-based representation has one decided advantage over traditional
# modeling: We can use BayBE's label encodings for the label parameters. For
# instance, when mixing small molecules, the
# [`SubstanceParameter`](baybe.parameters.substance.SubstanceParameter) can be used to
# smartly encode the slot labels, enabling the algorithm to perform a chemically-aware
# mixture optimization.

# In this example, we show how to design such a search space, including the various
# discrete constraints we need to impose. We simulate a situation where we want to mix
# up to three solvents, whose respective amounts must add up to 100.

# ```{admonition} Discrete vs. Continuous Modeling
# :class: important
# Here, we only use discrete parameters, although in principle the parameters
# corresponding to amounts could also be modeled as continuous numbers. However, this
# would imply that some of the constraints would have to act on both discrete and
# continuous parameters, which is not currently supported.
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

# Basic example settings:

SMOKE_TEST = "SMOKE_TEST" in os.environ
SUM_TOLERANCE = 0.1  # tolerance allowed to fulfill the sum constraints
RESOLUTION = 5 if SMOKE_TEST else 11  # resolution for discretizing the slot amounts

### Parameter Setup

# First, we create the parameters for the slot labels. Each slot offers a choice of
# four solvents:

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

# Next, we create the parameters representing the slot amounts:

slot1_amount = NumericalDiscreteParameter(
    name="Slot1_Amount", values=np.linspace(0, 100, RESOLUTION), tolerance=0.2
)
slot2_amount = NumericalDiscreteParameter(
    name="Slot2_Amount", values=np.linspace(0, 100, RESOLUTION), tolerance=0.2
)
slot3_amount = NumericalDiscreteParameter(
    name="Slot3_Amount", values=np.linspace(0, 100, RESOLUTION), tolerance=0.2
)

# We collect all parameters in a single list:

parameters = [
    slot1_label,
    slot2_label,
    slot3_label,
    slot1_amount,
    slot2_amount,
    slot3_amount,
]

### Constraint Setup

# For the sake of demonstration, we consider a scenario where we do *not* care about the
# order of addition of components to the mixture, which imposes two additional
# constraints.
#
# ```{admonition} Order of Addition
# :class: note
# Whether you need to impose the constraints for removing duplicates and imposing
# permutation invariance depends on your use case. If the order of addition is relevant
# to your mixture, the permutation invariance constraint should be discarded and one
# could further argue that adding the same substance multiple times should be allowed.
# ```

#### Duplicate Substances

# Assuming that the order of addition is irrelevant, there is no difference between
# having two slots with the same substance or having only one slot with the combined
# amounts. Thus, we want to make sure that there are no such duplicate label entries:

no_duplicates_constraint = DiscreteNoLabelDuplicatesConstraint(
    parameters=["Slot1_Label", "Slot2_Label", "Slot3_Label"]
)

#### Permutation Invariance

# Next, we need to take care of permutation invariance. If our order of addition does
# not matter, the result of interchanging any two slots does not alter the overall
# mixture, i.e. the mixture slots are are considered permutation-invariant.

# A complication with permutation invariance arises from the fact that we have not
# only a label per slot, but also a numerical amount. If this amount is zero, then the
# label of the slot becomes meaningless (i.e. which ingredient should be considered for the
# slot), because adding zero of it does not change the mixture. In BayBE, we call
# this a "dependency", i.e. the slot labels depend on the slot amounts and are only
# relevant if the amount satisfies some condition (in this case "amount > 0").

# The {class}`~baybe.constraints.discrete.DiscreteDependenciesConstraint` informs the
# {class}`~baybe.constraints.discrete.DiscretePermutationInvarianceConstraint` about
# these dependencies so that they are correctly included in the filtering process:

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

#### Substance Amounts

# Interpreting the slot amounts as percentages, we need to ensure that their total is
# always 100:

sum_constraint = DiscreteSumConstraint(
    parameters=["Slot1_Amount", "Slot2_Amount", "Slot3_Amount"],
    condition=ThresholdCondition(threshold=100, operator="=", tolerance=SUM_TOLERANCE),
)


# We store all constraints in a single list:

constraints = [perm_inv_constraint, sum_constraint, no_duplicates_constraint]


### Campaign Setup

# With all basic building blocks in place, we can now assemble our campaign:

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
