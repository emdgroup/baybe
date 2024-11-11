## Modeling a Mixture in Slot-Based Representation

### Terminology

# Modeling a mixture is possible in a non-traditional way by using a concept we refer to
# as a **slot**. A slot is represented through the combination of two parameters: one
# indicating the *amount* of a mixture ingredient, and another indicating the *type* of
# the ingredient (as a label) populating the slot. Unlike in [traditional mixture
# modeling](/examples/Mixtures/traditional.md), the total number of parameters is not
# determined by how many ingredient choices we have, but by the maximum number of slots
# we allow. For instance, if we want to design a mixture with *up to three* ingredients,
# we can do so by creating three slots represented by six parameters.

# A corresponding search space could look like this:
# | Slot1_Label | Slot1_Amount | Slot2_Label | Slot2_Amount | Slot3_Label | Slot3_Amount |
# |:------------|:-------------|:------------|:-------------|:------------|:-------------|
# | Solvent1    | 10           | Solvent5    | 20           | Solvent4    | 70           |
# | Solvent1    | 30           | Solvent8    | 40           | Solvent2    | 30           |
# | Solvent3    | 20           | Solvent1    | 35           | Solvent9    | 45           |
# | Solvent2    | 15           | Solvent3    | 40           | Solvent1    | 45           |

# The slot-based representation has one decided advantage over traditional
# modeling: We can use BayBE's label encodings for the label parameters. For
# instance, when mixing small molecules, the
# [`SubstanceParameter`](baybe.parameters.substance.SubstanceParameter) can be used to
# smartly encode the slot labels, enabling the algorithm to perform a chemically-aware
# mixture optimization.

# In this example, we show how to design such a search space, including the various
# discrete constraints we need to impose. We consider a situation where we want to mix
# up to three solvents, whose respective amounts must add up to 100.

# ```{admonition} Discrete vs. Continuous Modeling
# :class: important
#
# Here, we only use discrete parameters, although in principle the parameters
# corresponding to amounts could also be modeled as continuous numbers. However, this
# would imply that some of the constraints would have to act on both discrete and
# continuous parameters, which is not currently supported.
# ```

### Imports

import math

import numpy as np
import pandas as pd

from baybe.constraints import (
    DiscreteDependenciesConstraint,
    DiscreteNoLabelDuplicatesConstraint,
    DiscretePermutationInvarianceConstraint,
    DiscreteSumConstraint,
    ThresholdCondition,
)
from baybe.parameters import NumericalDiscreteParameter, SubstanceParameter
from baybe.searchspace import SubspaceDiscrete
from baybe.utils.dataframe import pretty_print_df

# Basic example settings:

SUM_TOLERANCE = 0.1  # tolerance allowed to fulfill the sum constraints
RESOLUTION = 5  # resolution for discretizing the slot amounts

### Parameter Setup

# First, we create the parameters for the slot labels. Each slot offers a choice of
# four solvents:

dict_solvents = {
    "water": "O",
    "ethanol": "CCO",
    "methanol": "CO",
    "acetone": "CC(=O)C",
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
# constraints: one for removing duplicates and one for imposing permutation invariance.
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
# amounts. Thus, we want to make sure that there are no such duplicate label entries,
# which can be achieved using a
# {class}`~baybe.constraints.discrete.DiscreteNoLabelDuplicatesConstraint`:

no_duplicates_constraint = DiscreteNoLabelDuplicatesConstraint(
    parameters=["Slot1_Label", "Slot2_Label", "Slot3_Label"]
)

#### Permutation Invariance

# Next, we need to take care of permutation invariance. If our order of addition does
# not matter, the result of interchanging any two slots does not alter the overall
# mixture, i.e. the mixture slots are considered permutation-invariant.

# A complication with permutation invariance arises from the fact that we do not only
# have a label per slot, but also a numerical amount. If this amount is zero, then the
# label of the slot becomes meaningless, because adding zero of the corresponding
# substance does not change the mixture. In BayBE, we call this a "dependency", i.e.
# the slot labels depend on the slot amounts and are only relevant if the amount
# satisfies some condition (in this case "amount > 0").

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


### Search Space Creation

# With all building blocks in place, we can now assemble our discrete space and inspect
# its configurations:


space = SubspaceDiscrete.from_product(parameters=parameters, constraints=constraints)
print(
    pretty_print_df(
        space.exp_rep,
        max_rows=len(space.exp_rep),
        max_columns=len(space.exp_rep.columns),
    )
)

# ````{admonition} Simplex Construction
# :class: tip
# In this example, we use the
# {meth}`~baybe.searchspace.discrete.SubspaceDiscrete.from_product` constructor in order
# to demonstrate the explicit creation of all involved constraints. However, for
# creating mixture representations, the
# {meth}`~baybe.searchspace.discrete.SubspaceDiscrete.from_simplex` constructor should
# generally be used. It takes care of the overall sum constraint already during search
# space creation, providing a more efficient path to the same result.
#
# The alternative in our case would look like:
# ```python
# space = SubspaceDiscrete.from_simplex(
#     max_sum=100.0,
#     boundary_only=True,
#     simplex_parameters=[slot1_amount, slot2_amount, slot3_amount],
#     product_parameters=[slot1_label, slot2_label, slot3_label],
#     constraints=[perm_inv_constraint, no_duplicates_constraint],
# )
# ```
# Note that {meth}`~baybe.searchspace.discrete.SubspaceDiscrete.from_simplex`
# inherently ensures the sum constraint, hence we do not pass it to `constraints`.
# ````


### Verification of Constraints

# Let us programmatically assert that all constraints are satisfied:

amounts = space.exp_rep[["Slot1_Amount", "Slot2_Amount", "Slot3_Amount"]]
labels = space.exp_rep[["Slot1_Label", "Slot2_Label", "Slot3_Label"]]
slots = space.exp_rep.apply(
    lambda row: pd.Series(
        [(row[f"Slot{k}_Label"], row[f"Slot{k}_Amount"]) for k in range(1, 4)]
    ),
    axis=1,
)

# * All amounts sum to 100:

n_wrong_sum = amounts.sum(axis=1).apply(lambda x: x - 100).abs().gt(SUM_TOLERANCE).sum()
assert n_wrong_sum == 0
print("Number of configurations whose amounts do not sum to 100: ", n_wrong_sum)


# * There are no duplicate slot labels:

n_duplicates = labels.nunique(axis=1).ne(3).sum()
assert n_duplicates == 0
print("Number of configurations with duplicate slot labels: ", n_duplicates)


# * There are no permutation-invariant configurations:

n_permute = slots.apply(frozenset, axis=1).duplicated().sum()
assert n_permute == 0
print("Number of permuted configurations: ", n_permute)

### Verification of Span

# Finally, we also assert if we have completely spanned the space of allowed
# configurations by comparing the numbers of unique `K`-solvent entries against their
# theoretical values.

# ```{admonition} Theoretical Span
# :class: info
#
# The number of possible `K`-solvent entries can be found by imagining the corresponding
# [traditional mixture representation](/examples/Mixtures/traditional.md) and solving a
# slightly more complex version of the ["stars and bars"
# problem](https://en.wikipedia.org/wiki/Stars_and_bars_(combinatorics)), where the
# number of non-empty bins is fixed. That is, we need to ask how many possible ways
# exist to distribute `N` items (= number of elemental steps for the amounts, in our
# case `RESOLUTION-1`) across `M` bins (= number of available solvents) if exactly
# `K` bins are non-empty (= number of solvents allowed in the mixture).
#
# There are `(M choose K)` ways to select the non-empty buckets. When distributing the
# `N` items, one item needs to go to each of the `K` buckets for it to be non-empty.
# The remaining `N - K` items can be freely distributed among the `K` buckets. The
# number of configurations for the latter is given by the "stars and bars" formula,
# which states that `X` indistinguishable items can be placed in `Y` distinguishable
# bins in `((X + Y -1) choose (Y - 1))` ways. Setting `X`=`N-K` and `Y`=`K` gives
# `((N - 1) choose (K - 1))`. Combined with the former count, we get the formula
# implemented in the helper function below.
# ```

# Helper function to compute the theoretical numbers:


def n_combinations(N: int, M: int, K: int) -> int:
    """Get number of ways to put `N` items into `M` bins yielding `K` non-empty bins."""
    return math.comb(M, K) * math.comb(N - 1, K - 1)


# Verify that the space is fully spanned:

for K in range(1, 4):
    n_combinations_expected = n_combinations(RESOLUTION - 1, len(dict_solvents), K)
    n_combinations_actual = (amounts != 0).sum(axis=1).eq(K).sum()
    assert n_combinations_expected == n_combinations_actual
    print(
        f"Number of unique {K}-solvent entries: "
        f"{n_combinations_actual} ({n_combinations_expected} expected)"
    )
