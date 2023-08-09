### Example for using custom constraints in discrete searchspaces

"""
This examples shows how a custom constraint can be created for a discrete searchspace.
That is, it shows how the user can define a constraint restricting the searchspace.
"""

# This example assumes some basic familiarity with using BayBE.
# We thus refer to [`baybe_object`](./../Basics/baybe_object.md) for a basic example.

#### Necessary imports for this example

import numpy as np
import pandas as pd
from baybe import BayBE

from baybe.constraints import CustomConstraint
from baybe.parameters import (
    CategoricalParameter,
    NumericalDiscreteParameter,
    SubstanceParameter,
)
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget, Objective
from baybe.utils import add_fake_results

### Experiment setup

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
solvent = SubstanceParameter("Solvent", data=dict_solvent, encoding="RDKIT")
speed = CategoricalParameter(
    "Speed", values=["very slow", "slow", "normal", "fast", "very fast"], encoding="INT"
)
temperature = NumericalDiscreteParameter(
    "Temperature", values=list(np.linspace(100, 200, 10)), tolerance=0.5
)
concentration = NumericalDiscreteParameter(
    "Concentration", values=[1, 2, 5, 10], tolerance=0.4
)

parameters = [solvent, speed, temperature, concentration]

### Creating the constraint

# The constraints are handled when creating the searchspace object.
# We thus need to define our constraint first as follows.


def custom_function(ser: pd.Series) -> bool:
    """
    This constraint implements a custom user-defined filter/validation functionality.
    """
    if ser.Solvent == "water":
        if ser.Temperature > 120 and ser.Concentration > 5:
            return False
        if ser.Temperature > 180 and ser.Concentration > 3:
            return False
    if ser.Solvent == "C3":
        if ser.Temperature < 150 and ser.Concentration > 3:
            return False
    return True


# We now initialize the `CustomConstraint` with all parameters this function should have access to.
constraint = CustomConstraint(
    parameters=["Concentration", "Solvent", "Temperature"], validator=custom_function
)

### Creating the searchspace and the objective

searchspace = SearchSpace.from_product(parameters=parameters, constraints=[constraint])

objective = Objective(
    mode="SINGLE", targets=[NumericalTarget(name="yield", mode="MAX")]
)

### Creating and printing the BayBE object

baybe_obj = BayBE(searchspace=searchspace, objective=objective)
print(baybe_obj)

### Manual verification of the constraint

# The following loop performs some recommendations and manually verifies the given constraints.
N_ITERATIONS = 5
for kIter in range(N_ITERATIONS):
    print(f"\n\n##### ITERATION {kIter+1} #####")

    print("### ASSERTS ###")
    print(
        "Number of entries with water, temp above 120 and concentration above 5:      ",
        (
            baybe_obj.searchspace.discrete.exp_rep["Concentration"].apply(
                lambda x: x > 5
            )
            & baybe_obj.searchspace.discrete.exp_rep["Temperature"].apply(
                lambda x: x > 120
            )
            & baybe_obj.searchspace.discrete.exp_rep["Solvent"].eq("water")
        ).sum(),
    )
    print(
        "Number of entries with water, temp above 180 and concentration above 3:      ",
        (
            baybe_obj.searchspace.discrete.exp_rep["Concentration"].apply(
                lambda x: x > 3
            )
            & baybe_obj.searchspace.discrete.exp_rep["Temperature"].apply(
                lambda x: x > 180
            )
            & baybe_obj.searchspace.discrete.exp_rep["Solvent"].eq("water")
        ).sum(),
    )
    print(
        "Number of entries with C3, temp above 180 and concentration above 3:         ",
        (
            baybe_obj.searchspace.discrete.exp_rep["Concentration"].apply(
                lambda x: x > 3
            )
            & baybe_obj.searchspace.discrete.exp_rep["Temperature"].apply(
                lambda x: x < 150
            )
            & baybe_obj.searchspace.discrete.exp_rep["Solvent"].eq("C3")
        ).sum(),
    )

    rec = baybe_obj.recommend(batch_quantity=5)
    add_fake_results(rec, baybe_obj)
    baybe_obj.add_measurements(rec)
