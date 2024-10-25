## Example for full simulation loop using the multi target mode for custom analytic functions

# This example shows how to use a multi target objective for a custom analytic function.
# It uses a desirability value to handle several targets.

# This example assumes basic familiarity with BayBE, custom test functions and multiple targets.
# For further details, we thus refer to
# - [`campaign`](./../Basics/campaign.md) for a more general and basic example,
# - [`custom_analytical`](./custom_analytical.md) for custom test functions, and
# - [`desirability`](./../Multi_Target/desirability.md) for multiple targets.

### Necessary imports for this example

import os

import numpy as np

from baybe import Campaign
from baybe.objectives import DesirabilityObjective
from baybe.parameters import NumericalDiscreteParameter
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget

### Parameters for a full simulation loop

# For the full simulation, we need to define some additional parameters.
# These are the number of Monte Carlo runs and the number of experiments to be conducted per run.

SMOKE_TEST = "SMOKE_TEST" in os.environ

N_MC_ITERATIONS = 2 if SMOKE_TEST else 5
N_DOE_ITERATIONS = 2 if SMOKE_TEST else 4
BATCH_SIZE = 1 if SMOKE_TEST else 2
DIMENSION = 4
BOUNDS = [(-2, 2), (-2, 2), (-2, 2), (-2, 2)]
POINTS_PER_DIM = 3 if SMOKE_TEST else 10


### Defining the test function

# See [`custom_analytical`](./custom_analytical.md) for details.


def sum_of_squares(*x: float) -> tuple[float, float]:
    """Calculate the sum of squares."""
    res = 0
    for y in x:
        res += y**2
    return res, 2 * res**2 - 1


### Creating the searchspace

# In this example, we construct a purely discrete space with 10 points per dimension.

parameters = [
    NumericalDiscreteParameter(
        name=f"x_{k+1}",
        values=list(np.linspace(*BOUNDS[k], POINTS_PER_DIM)),
        tolerance=0.01,
    )
    for k in range(DIMENSION)
]

searchspace = SearchSpace.from_product(parameters=parameters)


### Creating multiple target object

# The multi target mode is handled when creating the objective object.
# Thus, we first need to define the different targets.
# We use two targets here.
# The first target is maximized and the second target is minimized during the optimization process.

Target_1 = NumericalTarget(
    name="Target_1", mode="MAX", bounds=(0, 100), transformation="LINEAR"
)
Target_2 = NumericalTarget(
    name="Target_2", mode="MIN", bounds=(0, 100), transformation="LINEAR"
)


### Creating the objective object

# We collect the two targets in a list and use this list to construct the objective.

targets = [Target_1, Target_2]

objective = DesirabilityObjective(
    targets=targets,
    weights=[20, 30],
    scalarizer="MEAN",
)


### Constructing a campaign and performing the simulation loop

campaign = Campaign(searchspace=searchspace, objective=objective)

# We can now use the `simulate_scenarios` function to simulate a full experiment.

scenarios = {"BayBE": campaign}

results = simulate_scenarios(
    scenarios,
    sum_of_squares,
    batch_size=BATCH_SIZE,
    n_doe_iterations=N_DOE_ITERATIONS,
    n_mc_iterations=N_MC_ITERATIONS,
)

print(results)
