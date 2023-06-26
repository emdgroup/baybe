"""
This example shows how a dependency constraint can be created for a discrete
searchspace.
There are constraint that specifies dependencies between parameters. For instance some
parameters might only be relevant when another parameter has a certain value
(e.g. 'on'). All dependencies can be declared in a single constraint.
This example assumes that the reader is familiar with the basics of BayBE, and thus
does not explain the details of e.g. parameter creation. For additional explanation
on these aspects, we refer to the Basic examples.
"""

import numpy as np
from baybe.constraints import DependenciesConstraint, SubSelectionCondition

from baybe.core import BayBE
from baybe.parameters import Categorical, GenericSubstance, NumericDiscrete
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget, Objective
from baybe.utils import add_fake_results


# We begin by setting up some parameters for our experiments.
dict_solvent = {
    "water": "O",
    "C1": "C",
}
solvent = GenericSubstance(name="Solvent", data=dict_solvent, encoding="MORDRED")
switch1 = Categorical(name="Switch1", values=["on", "off"])
switch2 = Categorical(name="Switch2", values=["left", "right"])
fraction1 = NumericDiscrete(
    name="Fraction1", values=list(np.linspace(0, 100, 7)), tolerance=0.2
)
frame1 = Categorical(name="FrameA", values=["A", "B"])
frame2 = Categorical(name="FrameB", values=["A", "B"])

parameters = [solvent, switch1, switch2, fraction1, frame1, frame2]

# The constraints are handled when creating the searchspace object.
# We thus need to define our constraint now.
# The constraints can either be created jointly by constructing a single
# DependenciesConstraint or by having multiple constraints.
# This is demonstrated here by creating two BayBE objects

# This is the constraint modeling two Dependencies at once....
constraint_1 = DependenciesConstraint(
    parameters=["Switch1", "Switch2"],
    conditions=[
        SubSelectionCondition(selection=["on"]),
        SubSelectionCondition(selection=["right"]),
    ],
    affected_parameters=[["Solvent", "Fraction1"], ["FrameA", "FrameB"]],
)

# ... while these are two separate constraints that do the same when used together.
constraint_2 = DependenciesConstraint(
    parameters=["Switch1"],
    conditions=[SubSelectionCondition(selection=["on"])],
    affected_parameters=[["Solvent", "Fraction1"]],
)

constraint_3 = DependenciesConstraint(
    parameters=["Switch2"],
    conditions=[SubSelectionCondition(selection=["right"])],
    affected_parameters=[["FrameA", "FrameB"]],
)

# Create the search spaces with the corresponding lists of constraints
searchspace_1 = SearchSpace.create(parameters=parameters, constraints=[constraint_1])
searchspace_2 = SearchSpace.create(
    parameters=parameters, constraints=[constraint_2, constraint_3]
)

# Create the objective
objective = Objective(
    mode="SINGLE", targets=[NumericalTarget(name="Target_1", mode="MAX")]
)

# Put everything together
baybe_obj_1 = BayBE(searchspace=searchspace_1, objective=objective)
baybe_obj_2 = BayBE(searchspace=searchspace_2, objective=objective)

# Run some iterations for both constructed BayBE objects. During these iterations, we
# print some information about the parameters configurations that now exist.
N_ITERATIONS = 5
for kIter in range(N_ITERATIONS):
    print(f"\n##### Version1 ITERATION {kIter+1} #####")

    print("### ASSERTS ###")
    print(
        f"Number entries with both switches on "
        f"(expected {7*len(dict_solvent)*2*2}): ",
        (
            (baybe_obj_1.searchspace.discrete.exp_rep["Switch1"] == "on")
            & (baybe_obj_1.searchspace.discrete.exp_rep["Switch2"] == "right")
        ).sum(),
    )
    print(
        f"Number entries with Switch1 off " f"(expected {2*2}):       ",
        (
            (baybe_obj_1.searchspace.discrete.exp_rep["Switch1"] == "off")
            & (baybe_obj_1.searchspace.discrete.exp_rep["Switch2"] == "right")
        ).sum(),
    )
    print(
        f"Number entries with Switch2 off "
        f"(expected {7*len(dict_solvent)}):"
        f"      ",
        (
            (baybe_obj_1.searchspace.discrete.exp_rep["Switch1"] == "on")
            & (baybe_obj_1.searchspace.discrete.exp_rep["Switch2"] == "left")
        ).sum(),
    )
    print(
        "Number entries with both switches off (expected 1): ",
        (
            (baybe_obj_1.searchspace.discrete.exp_rep["Switch1"] == "off")
            & (baybe_obj_1.searchspace.discrete.exp_rep["Switch2"] == "left")
        ).sum(),
    )

    rec = baybe_obj_1.recommend(batch_quantity=5)
    add_fake_results(rec, baybe_obj_1)
    baybe_obj_1.add_measurements(rec)


for kIter in range(N_ITERATIONS):
    print(f"\n##### Version2 ITERATION {kIter+1} #####")

    print("### ASSERTS ###")
    print(
        f"Number entries with both switches on "
        f"(expected {7*len(dict_solvent)*2*2}): ",
        (
            (baybe_obj_2.searchspace.discrete.exp_rep["Switch1"] == "on")
            & (baybe_obj_2.searchspace.discrete.exp_rep["Switch2"] == "right")
        ).sum(),
    )
    print(
        f"Number entries with Switch1 off " f"(expected {2*2}):       ",
        (
            (baybe_obj_2.searchspace.discrete.exp_rep["Switch1"] == "off")
            & (baybe_obj_2.searchspace.discrete.exp_rep["Switch2"] == "right")
        ).sum(),
    )
    print(
        f"Number entries with Switch2 off "
        f"(expected {7*len(dict_solvent)}):"
        f"      ",
        (
            (baybe_obj_2.searchspace.discrete.exp_rep["Switch1"] == "on")
            & (baybe_obj_2.searchspace.discrete.exp_rep["Switch2"] == "left")
        ).sum(),
    )
    print(
        "Number entries with both switches off (expected 1): ",
        (
            (baybe_obj_2.searchspace.discrete.exp_rep["Switch1"] == "off")
            & (baybe_obj_2.searchspace.discrete.exp_rep["Switch2"] == "left")
        ).sum(),
    )

    rec = baybe_obj_2.recommend(batch_quantity=5)

    add_fake_results(rec, baybe_obj_2)
    baybe_obj_2.add_measurements(rec)
