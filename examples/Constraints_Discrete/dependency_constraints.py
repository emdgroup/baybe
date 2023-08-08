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

from baybe import BayBE
from baybe.constraints import DependenciesConstraint, SubSelectionCondition
from baybe.parameters import (
    CategoricalParameter,
    NumericalDiscreteParameter,
    SubstanceParameter,
)
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget, Objective
from baybe.utils import add_fake_results

# We begin by setting up some parameters for our experiments.
dict_solvent = {
    "water": "O",
    "C1": "C",
}
solvent = SubstanceParameter(name="Solvent", data=dict_solvent, encoding="MORDRED")
switch1 = CategoricalParameter(name="Switch1", values=["on", "off"])
switch2 = CategoricalParameter(name="Switch2", values=["left", "right"])
fraction1 = NumericalDiscreteParameter(
    name="Fraction1", values=list(np.linspace(0, 100, 7)), tolerance=0.2
)
frame1 = CategoricalParameter(name="FrameA", values=["A", "B"])
frame2 = CategoricalParameter(name="FrameB", values=["A", "B"])

parameters = [solvent, switch1, switch2, fraction1, frame1, frame2]

# The constraints are handled when creating the searchspace object.
# We thus need to define our constraint now.
# The constraints can either be created jointly by constructing a single
# DependenciesConstraint or by having multiple constraints.
# This is demonstrated here by creating two BayBE objects

# This is the constraint modeling two Dependencies. Multiple dependencies have to be
# included in a single constraint object
constraint = DependenciesConstraint(
    parameters=["Switch1", "Switch2"],
    conditions=[
        SubSelectionCondition(selection=["on"]),
        SubSelectionCondition(selection=["right"]),
    ],
    affected_parameters=[["Solvent", "Fraction1"], ["FrameA", "FrameB"]],
)


# Create the search spaces with the corresponding lists of constraints
searchspace = SearchSpace.from_product(parameters=parameters, constraints=[constraint])

# Create the objective
objective = Objective(
    mode="SINGLE", targets=[NumericalTarget(name="Target_1", mode="MAX")]
)

# Put everything together
baybe_obj = BayBE(searchspace=searchspace, objective=objective)

# Run some iterations. During these iterations, we
# print some information about the parameters configurations that now exist.
N_ITERATIONS = 5
for kIter in range(N_ITERATIONS):
    print(f"\n##### ITERATION {kIter+1} #####")

    print("### ASSERTS ###")
    print(
        f"Number entries with both switches on "
        f"(expected {7*len(dict_solvent)*2*2}): ",
        (
            (baybe_obj.searchspace.discrete.exp_rep["Switch1"] == "on")
            & (baybe_obj.searchspace.discrete.exp_rep["Switch2"] == "right")
        ).sum(),
    )
    print(
        f"Number entries with Switch1 off " f"(expected {2*2}):       ",
        (
            (baybe_obj.searchspace.discrete.exp_rep["Switch1"] == "off")
            & (baybe_obj.searchspace.discrete.exp_rep["Switch2"] == "right")
        ).sum(),
    )
    print(
        f"Number entries with Switch2 off "
        f"(expected {7*len(dict_solvent)}):"
        f"      ",
        (
            (baybe_obj.searchspace.discrete.exp_rep["Switch1"] == "on")
            & (baybe_obj.searchspace.discrete.exp_rep["Switch2"] == "left")
        ).sum(),
    )
    print(
        "Number entries with both switches off (expected 1): ",
        (
            (baybe_obj.searchspace.discrete.exp_rep["Switch1"] == "off")
            & (baybe_obj.searchspace.discrete.exp_rep["Switch2"] == "left")
        ).sum(),
    )

    rec = baybe_obj.recommend(batch_quantity=5)
    add_fake_results(rec, baybe_obj)
    baybe_obj.add_measurements(rec)
