"""
This examples shows how an exclusion constraint can be created for a discrete
searchspace, for instance if some parameter values are incompatible with certain values
of another parameter. It assumes that the reader is familiar with the basics of Baybe,
and thus does not explain the details of e.g. parameter creation. For additional
explanation on these aspects, we refer to the Basic examples.
"""
import numpy as np

from baybe.constraints import (
    ExcludeConstraint,
    SubSelectionCondition,
    ThresholdCondition,
)

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
    "C4": "CCCC",
    "C5": "CCCCC",
    "c6": "c1ccccc1",
    "C6": "CCCCCC",
}
solvent = GenericSubstance(name="Solvent", data=dict_solvent, encoding="RDKIT")
speed = Categorical(
    name="Speed",
    values=["very slow", "slow", "normal", "fast", "very fast"],
    encoding="INT",
)
temperature = NumericDiscrete(
    name="Temperature", values=list(np.linspace(100, 200, 15)), tolerance=0.4
)
pressure = NumericDiscrete(name="Pressure", values=[1, 2, 5, 10], tolerance=0.4)

parameters = [solvent, speed, temperature, pressure]

# This constraint simulates a situation where solvents C2 and C4 are not
# compatible with temperatures > 151 and should thus be excluded
constraint_1 = ExcludeConstraint(
    parameters=["Temperature", "Solvent"],
    combiner="AND",
    conditions=[
        ThresholdCondition(threshold=151, operator=">"),
        SubSelectionCondition(selection=["C4", "C2"]),
    ],
)
# This constraint simulates a situation where solvents C5 and C6 are not
# compatible with pressures > 5 and should thus be excluded
constraint_2 = ExcludeConstraint(
    parameters=["Pressure", "Solvent"],
    combiner="AND",
    conditions=[
        ThresholdCondition(threshold=5, operator=">"),
        SubSelectionCondition(selection=["C5", "C6"]),
    ],
)
# This constraint simulates a situation where pressures below 3 should never be
# combined with temperatures above 120
constraint_3 = ExcludeConstraint(
    parameters=["Pressure", "Temperature"],
    combiner="AND",
    conditions=[
        ThresholdCondition(threshold=3.0, operator="<"),
        ThresholdCondition(threshold=120.0, operator=">"),
    ],
)

# Creating the searchspace using the previously defined constraints
searchspace = SearchSpace.create(
    parameters=parameters, constraints=[constraint_1, constraint_2, constraint_3]
)

# Create the objective
objective = Objective(
    mode="SINGLE", targets=[NumericalTarget(name="Target_1", mode="MAX")]
)

# Put everything together
baybe_obj = BayBE(searchspace=searchspace, objective=objective)
print(baybe_obj)

# Perform some iterations and check that the searchspaces respects the given constraints
N_ITERATIONS = 3
for kIter in range(N_ITERATIONS):
    print(f"\n\n##### ITERATION {kIter+1} #####")

    print("### ASSERTS ###")
    print(
        "Number of entries with either Solvents C2 or C4 and a temperature above 151: ",
        (
            baybe_obj.searchspace.discrete.exp_rep["Temperature"].apply(
                lambda x: x > 151
            )
            & baybe_obj.searchspace.discrete.exp_rep["Solvent"].apply(
                lambda x: x in ["C2", "C4"]
            )
        ).sum(),
    )
    print(
        "Number of entries with either Solvents C5 or C6 and a pressure above 5:      ",
        (
            baybe_obj.searchspace.discrete.exp_rep["Pressure"].apply(lambda x: x > 5)
            & baybe_obj.searchspace.discrete.exp_rep["Solvent"].apply(
                lambda x: x in ["C5", "C6"]
            )
        ).sum(),
    )
    print(
        "Number of entries with pressure below 3 and temperature above 120:           ",
        (
            baybe_obj.searchspace.discrete.exp_rep["Pressure"].apply(lambda x: x < 3)
            & baybe_obj.searchspace.discrete.exp_rep["Temperature"].apply(
                lambda x: x > 120
            )
        ).sum(),
    )

    rec = baybe_obj.recommend(batch_quantity=5)
    add_fake_results(rec, baybe_obj)
    baybe_obj.add_measurements(rec)
