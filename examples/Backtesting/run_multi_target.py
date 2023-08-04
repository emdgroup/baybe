"""
This example shows how one can use a multi target mode for the objective when using a
custom analytic functions
It uses a desirability value to handle several targets.
It assumes that the reader is familiar with the basics of BayBE, as well as the basics
of using custom analytic functions and multiple targets.
We thus refer to the corresponding examples for more explanations on these aspects.
"""
from typing import Tuple

import numpy as np
from baybe.core import BayBE
from baybe.parameters import NumericalDiscreteParameter
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget, Objective


# We implement a simple sum of squares function with two outputs.
def sum_of_squares(*x: float) -> Tuple[float, float]:
    """
    Calculates the sum of squares.
    """
    res = 0
    for y in x:
        res += y**2
    return res, 2 * res**2 - 1


# For our actual experiment, we need to specify the number of dimension that we want
# to use as this is necessary to know for the creation of the parameters. The same is
# true for the bounds of the parameters which should be provided as a list of
# two-dimensional tuples.
DIMENSION = 4
BOUNDS = [(-2, 2), (-2, 2), (-2, 2), (-2, 2)]

# In this example, we construct a purely discrete space.
parameters = [
    NumericalDiscreteParameter(
        name=f"x_{k+1}",
        values=list(np.linspace(*BOUNDS[k], 15)),
        tolerance=0.01,
    )
    for k in range(DIMENSION)
]

searchspace = SearchSpace.from_product(parameters=parameters)


# TARGETS
# --------------------------------------------------------------------------------------


# The multi target mode is handled when creating the objective object
# Thus we first need to define the different targets

# We use two targets here.
# One which is maximized and one minimized during the optimization process.


Target_1 = NumericalTarget(
    name="Target_1", mode="MAX", bounds=(0, 100), bounds_transform_func="LINEAR"
)
Target_2 = NumericalTarget(
    name="Target_2", mode="MIN", bounds=(0, 100), bounds_transform_func="LINEAR"
)


# OBJECTIVE
# --------------------------------------------------------------------------------------


targets = [Target_1, Target_2]


objective = Objective(
    mode="DESIRABILITY",
    targets=targets,
    weights=[20, 30],
    combine_func="MEAN",
)


# We finally create the BayBE object and perform backtesting.

baybe_obj = BayBE(searchspace=searchspace, objective=objective)

scenarios = {"BayBE": baybe_obj}

results = simulate_scenarios(
    scenarios=scenarios,
    batch_quantity=2,
    n_exp_iterations=4,
    n_mc_iterations=2,
    lookup=sum_of_squares,
)

print(results)
