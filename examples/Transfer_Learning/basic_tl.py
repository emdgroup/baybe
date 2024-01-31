## Example for a basic example using the transfer learning capabilities

# This example shows how to use BayBE's transfer learning capabilities.
# We construct a campaign, give it access to data from a related, but different task
# and show how this additional information boosts performance.

### Necessary imports for this example

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from botorch.test_functions.synthetic import Hartmann

from baybe import Campaign
from baybe.objective import Objective
from baybe.parameters import NumericalDiscreteParameter, TaskParameter
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget
from baybe.utils.botorch_wrapper import botorch_function_wrapper

### Parameters for the example

# These parameters are used to set up the simulation loop. They define the dimension of
# the test problem, the number of Monte Carlo runs, the number of DoE iteration, the
# batch quantity and the number of points per dimension.
# Note that this example checks whether the environment variable `SMOKE_TEST` is set.
# If this is the case, lower values are used for most of the parameters to significantly
# accelerate the calculation.


SMOKE_TEST = "SMOKE_TEST" in os.environ
DIMENSION = 3
N_MC_ITERATIONS = 5 if SMOKE_TEST else 75
N_DOE_ITERATIONS = 5 if SMOKE_TEST else 10
POINTS_PER_DIM = 5 if SMOKE_TEST else 5

### Defining the tasks

# We use the Hartmann test function from botorch.
# In this example, we assume that we have data available from the negated version of
# this function, while we use the original as the task that we want to optimize.
hartmann = botorch_function_wrapper(Hartmann(dim=DIMENSION))
negative_hartmann = botorch_function_wrapper(Hartmann(dim=DIMENSION, negate=True))

# We define a dictionary mapping the names of the functions to the objects since this
# will be useful later.
test_functions = {
    "Hartmann": hartmann,
    "Negative_Hartmann": negative_hartmann,
}

### Creating the searchspace and the objective

# The bounds of the search space are defined by the test function.

BOUNDS = Hartmann().bounds

# We define one numerical discrete parameters per dimension, as well as a `TaskParameter`.
# This parameter contains the information about the different tasks that we have in this
# transfer learning example.
parameters = [
    NumericalDiscreteParameter(
        name=f"x{k}",
        values=list(np.linspace(BOUNDS[0, k], BOUNDS[1, k], POINTS_PER_DIM)),
    )
    for k in range(DIMENSION)
]

# Since we have two different tasks but only want to optimize for one of them, we use
# the `active_values` keyword to define the parameter for which we want to obtain
# recommendations.
# For more details on this keyword, we refer to the [userguide](./../../userguide/transfer_learning.md).

task_param = TaskParameter(
    name="Function",
    values=("Hartmann", "Negative_Hartmann"),
    active_values=["Hartmann"],
)
parameters.append(task_param)

# We now create the searchspace, objective and campaign.

searchspace = SearchSpace.from_product(parameters=parameters)
objective = Objective(
    mode="SINGLE", targets=[NumericalTarget(name="Target", mode="MIN")]
)


### Generating lookup tables

# We generate two lookup tables for the values of the two Hartmann functions.

grid = np.meshgrid(*[parameters[k].values for k in range(DIMENSION)])

lookup_functions = {}

for function in test_functions:
    hartmann_function = test_functions[function]  # Get actual function
    lookup_tmp = pd.DataFrame({f"x{k}": grid[k].flatten() for k in range(DIMENSION)})
    lookup_tmp["Target"] = lookup_tmp.apply(hartmann_function, axis=1)
    lookup_tmp["Function"] = function
    lookup_functions[function] = lookup_tmp

### Performing the simulation loop

# We now perform the simulation for different percentages of available data.
# The results are concatenated and then plotted.

results = []
lookup_training = lookup_functions["Negative_Hartmann"]
for p in (0.01, 0.02, 0.05, 0.08, 0.2):
    campaign = Campaign(searchspace=searchspace, objective=objective)
    initial_data = [lookup_training.sample(frac=p) for _ in range(N_MC_ITERATIONS)]
    tmp_results = simulate_scenarios(
        {f"{100*p}": campaign},
        lookup_functions["Hartmann"],
        initial_data=initial_data,
        batch_size=1,
        n_doe_iterations=N_DOE_ITERATIONS,
    )
    results.append(tmp_results)

# For comparison, we also optimize the function without using any initial data.
tmp_results = simulate_scenarios(
    {"0": Campaign(searchspace=searchspace, objective=objective)},
    lookup_functions["Hartmann"],
    batch_size=1,
    n_doe_iterations=N_DOE_ITERATIONS,
    n_mc_iterations=N_MC_ITERATIONS,
)

results = pd.concat([tmp_results, *results])

# The following code creates up to 3 different plots for the results, called "light", "dark", and "check".
# The "light" and "dark" plots are meant to be included in the documentation.
# They are only saved if they do not already exist.
# The "check" plot is always created and thus overrides an already existing corresponding plot.

themes = json.load(open("plotting_themes.json"))
for theme in themes:
    font_scale, rc_params = themes[theme]["font_scale"], themes[theme]["rc_params"]
    sns.set_theme(style="ticks", font_scale=font_scale, rc=rc_params)
    sns.lineplot(data=results, x="Num_Experiments", y="Target_CumBest", hue="Scenario")

    if not (path := Path(sys.path[0], f"botorch_analytical_{theme}.svg")).exists() or (
        theme == "check"
    ):
        plt.savefig(
            path,
            format="svg",
            transparent=True,
        )
    if theme == "check":
        plt.show()
    plt.clf()
