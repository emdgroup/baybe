### Example for using a sum constraint in a discrete searchspace

"""
Example for imposing sum constraints for discrete parameters.
The constraints simulate a situation where we want to mix up to three solvents.
However, their respective fractions need to sum up to 100.
Also, the solvents should never be chosen twice.
"""

# This example assumes some basic familiarity with using BayBE.
# We thus refer to [`baybe_object`](./../Basics/baybe_object.md) for a basic example.

#### Necessary imports for this example

import math

import numpy as np

from baybe.constraints import (
    DependenciesConstraint,
    NoLabelDuplicatesConstraint,
    PermutationInvarianceConstraint,
    SumConstraint,
    ThresholdCondition,
)

from baybe.core import BayBE
from baybe.parameters import GenericSubstance, NumericDiscrete
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget, Objective
from baybe.utils import add_fake_results

### Experiment setup

# This parameter denotes the tolerance with regard to the calculation of the sum.
SUM_TOLERANCE = 1.0

dict_solvents = {
    "water": "O",
    "C1": "C",
    "C2": "CC",
    "C3": "CCC",
}
solvent1 = GenericSubstance(name="Solvent1", data=dict_solvents, encoding="MORDRED")
solvent2 = GenericSubstance(name="Solvent2", data=dict_solvents, encoding="MORDRED")
solvent3 = GenericSubstance(name="Solvent3", data=dict_solvents, encoding="MORDRED")

# Parameters for representing the fraction.
fraction1 = NumericDiscrete(
    name="Fraction1", values=list(np.linspace(0, 100, 12)), tolerance=0.2
)
fraction2 = NumericDiscrete(
    name="Fraction2", values=list(np.linspace(0, 100, 12)), tolerance=0.2
)
fraction3 = NumericDiscrete(
    name="Fraction3", values=list(np.linspace(0, 100, 12)), tolerance=0.2
)

parameters = [solvent1, solvent2, solvent3, fraction1, fraction2, fraction3]

### Creating the constraint

# Since the constraints are required for the creation of the searchspace, we create them next.
# Note that we need a `PermutationInvarianceConstraint` here.
# The reason is that constraints are normally applied in a specific order.
# However, the fractions should be invariant under permutations.
# We thus require an explicit constraint for this.
perm_inv_constraint = PermutationInvarianceConstraint(
    parameters=["Solvent1", "Solvent2", "Solvent3"],
    dependencies=DependenciesConstraint(
        parameters=["Fraction1", "Fraction2", "Fraction3"],
        conditions=[
            ThresholdCondition(threshold=0.0, operator=">"),
            ThresholdCondition(threshold=0.0, operator=">"),
            ThresholdCondition(threshold=0.0, operator=">"),
        ],
        affected_parameters=[["Solvent1"], ["Solvent2"], ["Solvent3"]],
    ),
)

# This is now the actual sum constraint
sum_constraint = SumConstraint(
    parameters=["Fraction1", "Fraction2", "Fraction3"],
    condition=ThresholdCondition(threshold=100, operator="=", tolerance=SUM_TOLERANCE),
)

# The permutation invariance might create duplciate labels.
# We thus include a constraint to remove them.
no_duplicates_constraint = NoLabelDuplicatesConstraint(
    parameters=["Solvent1", "Solvent2", "Solvent3"]
)

constraints = [perm_inv_constraint, sum_constraint, no_duplicates_constraint]

### Creating the searchspace and the objective

searchspace = SearchSpace.from_product(parameters=parameters, constraints=constraints)

objective = Objective(
    mode="SINGLE", targets=[NumericalTarget(name="Target_1", mode="MAX")]
)

### Creating and printing the BayBE object

baybe_obj = BayBE(searchspace=searchspace, objective=objective)
print(baybe_obj)

### Manual verification of the constraint

# The following loop performs some recommendations and manually verifies the given constraints.

N_ITERATIONS = 3
for kIter in range(N_ITERATIONS):
    print(f"\n##### ITERATION {kIter+1} #####")

    print("### ASSERTS ###")
    print(
        "No. of searchspace entries where fractions do not sum to 100.0:      ",
        baybe_obj.searchspace.discrete.exp_rep[["Fraction1", "Fraction2", "Fraction3"]]
        .sum(axis=1)
        .apply(lambda x: x - 100.0)
        .abs()
        .gt(SUM_TOLERANCE)
        .sum(),
    )
    print(
        "No. of searchspace entries that have duplicate solvent labels:       ",
        baybe_obj.searchspace.discrete.exp_rep[["Solvent1", "Solvent2", "Solvent3"]]
        .nunique(axis=1)
        .ne(3)
        .sum(),
    )
    print(
        "No. of searchspace entries with permutation-invariant combinations:  ",
        baybe_obj.searchspace.discrete.exp_rep[["Solvent1", "Solvent2", "Solvent3"]]
        .apply(frozenset, axis=1)
        .to_frame()
        .join(
            baybe_obj.searchspace.discrete.exp_rep[
                ["Fraction1", "Fraction2", "Fraction3"]
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
            baybe_obj.searchspace.discrete.exp_rep[
                ["Fraction1", "Fraction2", "Fraction3"]
            ]
            == 0.0
        )
        .sum(axis=1)
        .eq(2)
        .sum(),
    )
    print(
        f"No. of unique 2-solvent entries (exp."
        f" {math.comb(len(dict_solvents), 2)*(12-2)})",
        (
            baybe_obj.searchspace.discrete.exp_rep[
                ["Fraction1", "Fraction2", "Fraction3"]
            ]
            == 0.0
        )
        .sum(axis=1)
        .eq(1)
        .sum(),
    )
    print(
        f"No. of unique 3-solvent entries (exp."
        f" {math.comb(len(dict_solvents), 3)*((12-3)*(12-2))//2})",
        (
            baybe_obj.searchspace.discrete.exp_rep[
                ["Fraction1", "Fraction2", "Fraction3"]
            ]
            == 0.0
        )
        .sum(axis=1)
        .eq(0)
        .sum(),
    )

    rec = baybe_obj.recommend(batch_quantity=5)
    add_fake_results(rec, baybe_obj)
    baybe_obj.add_measurements(rec)
