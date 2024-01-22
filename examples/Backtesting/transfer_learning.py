### Example for a simulation loop using the transfer learning capabilities

# This example shows how to use BayBE's transfer learning capabilities in a backtesting
# scenario. We also compare the results obtained with transfer learning to results
# obtained without transfer learning.

# This example assumes some basic familiarity with using BayBE and the lookup
# functionality functions in discrete search spaces.
# We thus refer to
# 1. [`campaign`](./../Basics/campaign.md) for a basic example on how to use BayBE and
# 2. [`full_lookup`](./full_lookup.md) for details on the lookup functionality.

#### Necessary imports for this example

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from botorch.test_functions.synthetic import (
    ConstrainedHartmann,
    ConstrainedHartmannSmooth,
    Hartmann,
)

from baybe import Campaign
from baybe.objective import Objective
from baybe.parameters import NumericalDiscreteParameter, TaskParameter
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios, simulate_transfer_learning
from baybe.targets import NumericalTarget
from baybe.utils import botorch_function_wrapper

#### Parameters for a full simulation loop

# For the full simulation, we need to define some additional parameters.
# These are the number of Monte Carlo runs, the number of experiments to be conducted per run
# and the batch quantity.
# Also, since the test function we use here is defined for several dimension, we choose
# a dimension.

DIMENSION = 3
N_MC_ITERATIONS = 6
N_DOE_ITERATIONS = 5
BATCH_QUANTITY = 1

#### Defining the tasks

# We use three variants of the Hartmann Test function from botorch.
# These are the general Hartmann function, a constrained variant, and tow constrained
# variants. For details, see [here](https://botorch.org/v/0.1.4/api/test_functions.html).
hartmann = botorch_function_wrapper(Hartmann(dim=DIMENSION))
c_hartmann = botorch_function_wrapper(ConstrainedHartmann(dim=DIMENSION))
cs_hartmann = botorch_function_wrapper(ConstrainedHartmannSmooth(dim=DIMENSION))

# We define a dictionary mapping the names of the functions to the objects since this
# will be useful later.
test_functions = {
    "Hartmann": hartmann,
    "CHartmann": c_hartmann,
    "CSHartmann": cs_hartmann,
}

#### Creating the searchspace and the objective

# The parameter `POINTS_PER_DIM` controls the number of points per dimension.
# Note that the searchspace will have `POINTS_PER_DIM**DIMENSION` many points.
# The bounds are defined by the test function.
POINTS_PER_DIM = 6
BOUNDS = Hartmann().bounds

# We define one numerical discrete parameters per dimension, as well as a ``TaskParameter``.
# This parameter contains the information about the different tasks that we have in this
# transfer learning example.
parameters = [
    NumericalDiscreteParameter(
        name=f"x{k}",
        values=list(np.linspace(BOUNDS[0, k], BOUNDS[1, k], POINTS_PER_DIM)),
    )
    for k in range(DIMENSION)
]

# In this backtesting example, the goal is to use transfer learning to learn all three
# test functions and using the available data for all three of them. Thus, we do not
# declare any `active_values` here.
# For more details on these, we refer to the [userguide](./../../userguide/transfer_learning.md).

task_param = TaskParameter(
    name="Function", values=("Hartmann", "CHartmann", "CSHartmann")
)
parameters.append(task_param)

# We now create the searchspace, objective and campaign.

searchspace = SearchSpace.from_product(parameters=parameters)
objective = Objective(
    mode="SINGLE", targets=[NumericalTarget(name="Target", mode="MIN")]
)
campaign = Campaign(searchspace=searchspace, objective=objective)

#### Generating the lookup table

# The lookup table for the parameters and their values need to contain one column per
# parameter, including one for the ``TaskParameter`` and one column for the target value.
# The table is created by constructing one dataframe per task and concatenating them.

grid = np.meshgrid(*[parameters[k].values for k in range(DIMENSION)])

lookup = pd.DataFrame()

for function in test_functions:
    hartmann_function = test_functions[function]  # Get actual function
    lookup_tmp = pd.DataFrame({f"x{k}": grid[k].flatten() for k in range(DIMENSION)})
    lookup_tmp["Target"] = lookup_tmp.apply(hartmann_function, axis=1)
    lookup_tmp["Function"] = function
    lookup = pd.concat([lookup, lookup_tmp], ignore_index=True)


#### Performing the simulation loop

# We can now use the `simulate_transfer_learning` function to simulate a full experiment.

results = simulate_transfer_learning(
    campaign,
    lookup,
    batch_quantity=BATCH_QUANTITY,
    n_doe_iterations=N_DOE_ITERATIONS,
    n_mc_iterations=N_MC_ITERATIONS,
)

# To showcase the improvement, we also optimize the same tasks independently and use
# the `simulate_scenarios` function.

parameters_no_tl = parameters[:-1]  # We need new task parameters
scenarios = {}  # Dictionary of all scenarios
# Iterate over all available test functions to create one campaign for each function
for function in test_functions:
    hartmann_function = test_functions[function]  # Get current test function
    task_parameter = TaskParameter(  # Create TaskParameter
        name="Function",
        values=("Hartmann", "CHartmann", "CSHartmann"),
        active_values=[function],
    )
    campaign = Campaign(
        searchspace=SearchSpace.from_product(parameters_no_tl + [task_parameter]),
        objective=objective,
    )
    scenarios[f"{function}_no_TL"] = campaign

results_no_tl = simulate_scenarios(
    scenarios,
    lookup,
    batch_quantity=BATCH_QUANTITY,
    n_doe_iterations=N_DOE_ITERATIONS,
    n_mc_iterations=N_MC_ITERATIONS,
)
results = pd.concat([results, results_no_tl])


# We concatenate all the results and show them in a single plot.

sns.lineplot(data=results, x="Num_Experiments", y="Target_CumBest", hue="Scenario")
plt.gcf().set_size_inches(24, 8)
plt.savefig("./run_transfer_learning.png")
