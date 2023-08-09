### Example for using dependency constraints in discrete searchspaces

"""
This example shows how a dependency constraint can be created for a discrete searchspace.
For instance, some parameters might only be relevant when another parameter has a certain value.
All dependencies have to be declared in a single constraint.
"""

# This example assumes some basic familiarity with using BayBE.
# We thus refer to [`baybe_object`](./../Basics/baybe_object.md) for a basic example.

#### Necessary imports for this example

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

#### Experiment setup

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

#### Creating the constraints

# The constraints are handled when creating the searchspace object.
# It is thus necessary to define it before the searchspace creation.
# Note that multiple dependencies have to be included in a single constraint object.
constraint = DependenciesConstraint(
    parameters=["Switch1", "Switch2"],
    conditions=[
        SubSelectionCondition(selection=["on"]),
        SubSelectionCondition(selection=["right"]),
    ],
    affected_parameters=[["Solvent", "Fraction1"], ["FrameA", "FrameB"]],
)

#### Creating the searchspace and the objective

searchspace = SearchSpace.from_product(parameters=parameters, constraints=[constraint])

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
