## Example for a basic example using the transfer learning capabilities

# This example shows how to use BayBE's transfer learning capabilities.
# We construct a campaign, give it access to data from a related, but different task
# and show how this additional information boosts performance.

### Necessary imports for this example

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

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
BATCH_SIZE = 1
N_MC_ITERATIONS = 5 if SMOKE_TEST else 75
N_DOE_ITERATIONS = 5 if SMOKE_TEST else 10
POINTS_PER_DIM = 5 if SMOKE_TEST else 5

### Defining the tasks

# We use the Hartmann test function from botorch.
# In this example, we assume that we have data available from the negated version of
# this function, while we use the original as the task that we want to optimize.
# We define a dictionary mapping the names of the functions to the objects since this
# will be useful later.
test_functions = {
    "Hartmann": botorch_function_wrapper(Hartmann(dim=DIMENSION)),
    "Negative_Hartmann": botorch_function_wrapper(Hartmann(dim=DIMENSION, negate=True)),
}

### Creating the searchspace and the objective

# The bounds of the search space are defined by the test function.

BOUNDS = Hartmann(dim=DIMENSION).bounds

# We define one numerical discrete parameters per dimension, as well as a `TaskParameter`.
# This parameter contains the information about the different tasks that we have in this
# transfer learning example.
discrete_params = [
    NumericalDiscreteParameter(
        name=f"x{d}",
        values=np.linspace(lower, upper, POINTS_PER_DIM),
    )
    for d, (lower, upper) in enumerate(BOUNDS.T)
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

# We now create the searchspace, objective and campaign.

parameters = [*discrete_params, task_param]
searchspace = SearchSpace.from_product(parameters=parameters)
objective = Objective(
    mode="SINGLE", targets=[NumericalTarget(name="Target", mode="MIN")]
)


### Generating lookup tables

# We generate two lookup tables for the values of the two Hartmann functions.

grid = np.meshgrid(*[p.values for p in discrete_params])

lookups: Dict[str, pd.DataFrame] = {}
for function_name, function in test_functions.items():
    lookup = pd.DataFrame({f"x{d}": grid_d.ravel() for d, grid_d in enumerate(grid)})
    lookup["Target"] = lookup.apply(function, axis=1)
    lookup["Function"] = function_name
    lookups[function_name] = lookup
lookup_training_task = lookups["Negative_Hartmann"]
lookup_test_task = lookups["Hartmann"]

### Performing the simulation loop

# We now perform the simulation for different percentages of available data.
# The results are concatenated and then plotted.

results: List[pd.DataFrame] = []
for p in (0.01, 0.02, 0.05, 0.08, 0.2):
    campaign = Campaign(searchspace=searchspace, objective=objective)
    initial_data = [lookup_training_task.sample(frac=p) for _ in range(N_MC_ITERATIONS)]
    result_fraction = simulate_scenarios(
        {f"{100*p}": campaign},
        lookup_test_task,
        initial_data=initial_data,
        batch_size=BATCH_SIZE,
        n_doe_iterations=N_DOE_ITERATIONS,
    )
    results.append(result_fraction)

# For comparison, we also optimize the function without using any initial data.
result_fraction = simulate_scenarios(
    {"0.0": Campaign(searchspace=searchspace, objective=objective)},
    lookup_test_task,
    batch_size=BATCH_SIZE,
    n_doe_iterations=N_DOE_ITERATIONS,
    n_mc_iterations=N_MC_ITERATIONS,
)

results = pd.concat([result_fraction, *results])

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
