"""
This examples shows how an exclusion constraint can be created for a discrete
searchspace using products and sums. It assumes that the reader is familiar with the
basics of BayBE, and thus does not explain the details of e.g. parameter creation.
For additional explanation on these aspects, we refer to the Basic examples.
"""
import numpy as np

from baybe.constraints import ProductConstraint, SumConstraint, ThresholdCondition

from baybe.core import BayBE
from baybe.parameters import Categorical, GenericSubstance, NumericDiscrete
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget, Objective
from baybe.utils import add_fake_results


# We begin by setting up some parameters for our experiments.
dict_solvent = {
    "water": "O",
    "C1": "C",
    "C2": "CC",
    "C3": "CCC",
}
solvent = GenericSubstance(name="Solvent", data=dict_solvent, encoding="RDKIT")
speed = Categorical(name="Speed", values=["slow", "normal", "fast"], encoding="INT")
num_parameter_1 = NumericDiscrete(
    name="NumParameter1", values=list(np.linspace(0, 100, 7)), tolerance=0.5
)
num_parameter_2 = NumericDiscrete(
    name="NumParameter2", values=list(np.linspace(0, 100, 7)), tolerance=0.5
)
num_parameter_3 = NumericDiscrete(
    name="NumParameter3", values=list(np.linspace(0, 100, 7)), tolerance=0.5
)
num_parameter_4 = NumericDiscrete(
    name="NumParameter4", values=list(np.linspace(0, 100, 7)), tolerance=0.5
)
num_parameter_5 = NumericDiscrete(
    name="NumParameter5", values=list(np.linspace(0, 100, 7)), tolerance=0.5
)
num_parameter_6 = NumericDiscrete(
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

# Before creating the searchspace, we create the constraints

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

searchspace = SearchSpace.create(parameters=parameters, constraints=constraints)

# Create the objective
objective = Objective(
    mode="SINGLE", targets=[NumericalTarget(name="Target_1", mode="MAX")]
)

# Create BayBE object, add fake results and print what happens to internal data
baybe_obj = BayBE(searchspace=searchspace, objective=objective)
print(baybe_obj)

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
