### Example for using exlusion constraints incorporating sums and products

"""
This examples demonstrates an exclusion constraint using products and sums.
"""

# This example assumes some basic familiarity with using BayBE.
# We thus refer to [`baybe_object`](./../Basics/baybe_object.md) for a basic example.

#### Necessary imports for this example

import numpy as np
from baybe import BayBE

from baybe.constraints import ProductConstraint, SumConstraint, ThresholdCondition
from baybe.parameters import (
    CategoricalParameter,
    NumericalDiscreteParameter,
    SubstanceParameter,
)
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget, Objective
from baybe.utils import add_fake_results

#### Experiment setup

dict_solvent = {
    "water": "O",
    "C1": "C",
    "C2": "CC",
    "C3": "CCC",
}
solvent = SubstanceParameter(name="Solvent", data=dict_solvent, encoding="RDKIT")
speed = CategoricalParameter(
    name="Speed", values=["slow", "normal", "fast"], encoding="INT"
)
num_parameter_1 = NumericalDiscreteParameter(
    name="NumParameter1", values=list(np.linspace(0, 100, 7)), tolerance=0.5
)
num_parameter_2 = NumericalDiscreteParameter(
    name="NumParameter2", values=list(np.linspace(0, 100, 7)), tolerance=0.5
)
num_parameter_3 = NumericalDiscreteParameter(
    name="NumParameter3", values=list(np.linspace(0, 100, 7)), tolerance=0.5
)
num_parameter_4 = NumericalDiscreteParameter(
    name="NumParameter4", values=list(np.linspace(0, 100, 7)), tolerance=0.5
)
num_parameter_5 = NumericalDiscreteParameter(
    name="NumParameter5", values=list(np.linspace(0, 100, 7)), tolerance=0.5
)
num_parameter_6 = NumericalDiscreteParameter(
    name="NumParameter6", values=list(np.linspace(0, 100, 7)), tolerance=0.5
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

#### Creating the constraints

# Constraints are used when creating the searchspace object.
# Thus, they need to be defined prior to the searchspace creation.

sum_constraint_1 = SumConstraint(
    parameters=["NumParameter1", "NumParameter2"],
    condition=ThresholdCondition(threshold=150.0, operator="<="),
)
sum_constraint_2 = SumConstraint(
    parameters=["NumParameter5", "NumParameter6"],
    condition=ThresholdCondition(threshold=100, operator="=", tolerance=1.0),
)
prod_constraint = ProductConstraint(
    parameters=["NumParameter3", "NumParameter4"],
    condition=ThresholdCondition(threshold=30, operator=">="),
)

constraints = [sum_constraint_1, sum_constraint_2, prod_constraint]

#### Creating the searchspace and the objective

searchspace = SearchSpace.from_product(parameters=parameters, constraints=constraints)

objective = Objective(
    mode="SINGLE", targets=[NumericalTarget(name="Target_1", mode="MAX")]
)

#### Creating and printing the BayBE object

baybe_obj = BayBE(searchspace=searchspace, objective=objective)
print(baybe_obj)

#### Manual verification of the constraints

# The following loop performs some recommendations and manually verifies the given constraints.

N_ITERATIONS = 5
for kIter in range(N_ITERATIONS):
    print(f"\n\n##### ITERATION {kIter+1} #####")

    print("### ASSERTS ###")
    print(
        "Number of entries with 1,2-sum above 150:      ",
        (
            baybe_obj.searchspace.discrete.exp_rep[
                ["NumParameter1", "NumParameter2"]
            ].sum(axis=1)
            > 150.0
        ).sum(),
    )
    print(
        "Number of entries with 3,4-product under 30:   ",
        (
            baybe_obj.searchspace.discrete.exp_rep[
                ["NumParameter3", "NumParameter4"]
            ].prod(axis=1)
            < 30
        ).sum(),
    )
    print(
        "Number of entries with 5,6-sum unequal to 100: ",
        baybe_obj.searchspace.discrete.exp_rep[["NumParameter5", "NumParameter6"]]
        .sum(axis=1)
        .apply(lambda x: x - 100.0)
        .abs()
        .gt(0.01)
        .sum(),
    )

    rec = baybe_obj.recommend(batch_quantity=5)
    add_fake_results(rec, baybe_obj)
    baybe_obj.add_measurements(rec)
