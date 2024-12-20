## Backtesting

# This example demonstrates the use of the
# [`simulate_transfer_learning`](baybe.simulation.transfer_learning.simulate_transfer_learning)
# function to learn across tasks:
# * We construct a campaign,
# * define two related test functions,
# * use the data from the first function to train the second,
# * and vice versa

### Imports

import os

import numpy as np
import pandas as pd
import seaborn as sns
from botorch.test_functions.synthetic import Hartmann
from torch import Tensor

from baybe import Campaign
from baybe.parameters import NumericalDiscreteParameter, TaskParameter
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios, simulate_transfer_learning
from baybe.targets import NumericalTarget
from baybe.utils.dataframe import arrays_to_dataframes
from baybe.utils.plotting import create_example_plots

### Settings

# The following settings are used to set up the problem:

SMOKE_TEST = "SMOKE_TEST" in os.environ  # reduce the problem complexity in CI pipelines
DIMENSION = 3  # input dimensionality of the test function
BATCH_SIZE = 1  # batch size of recommendations per DOE iteration
N_MC_ITERATIONS = 2 if SMOKE_TEST else 50  # number of Monte Carlo runs
N_DOE_ITERATIONS = 2 if SMOKE_TEST else 10  # number of DOE iterations
POINTS_PER_DIM = 3 if SMOKE_TEST else 7  # number of grid points per input dimension


### Creating the Optimization Objective

# The test functions each have a single output that is to be minimized.
# The corresponding [Objective](baybe.objective.Objective)
# is created as follows:

target = NumericalTarget(name="Target", mode="MIN")
objective = target.to_objective()

### Creating the Search Space

# This example uses the [Hartmann Function](https://botorch.org/api/test_functions.html#botorch.test_functions.synthetic.Hartmann)
# as implemented by `botorch`.
# The bounds of the search space are dictated by the test function and can be extracted
# from the function itself.

BOUNDS = Hartmann(dim=DIMENSION).bounds

# First, we define one
# [NumericalDiscreteParameter](baybe.parameters.numerical.NumericalDiscreteParameter)
# per input dimension of the test function:

discrete_params = [
    NumericalDiscreteParameter(
        name=f"x{d}",
        values=np.linspace(lower, upper, POINTS_PER_DIM),
    )
    for d, (lower, upper) in enumerate(BOUNDS.T)
]


# Next, we define a
# [TaskParameter](baybe.parameters.categorical.TaskParameter) to encode the task context,
# which allows the model to establish a relationship between the training data and
# the data collected during the optimization process.
# Since we perform a cross training here, we do not specify any `active_values`.

task_param = TaskParameter(
    name="Function",
    values=["Hartmann", "Shifted"],
)

# With the parameters at hand, we can now create our search space.

parameters = [*discrete_params, task_param]
searchspace = SearchSpace.from_product(parameters=parameters)

### Defining the Tasks

# To demonstrate the transfer learning mechanism, we consider the problem of optimizing
# the Hartmann function using training data from a shifted, scaled and noisy version
# and vice versa. The used model is of course not aware of this relationship but
# needs to infer it from the data gathered during the optimization process.

wrapper = arrays_to_dataframes(
    [p.name for p in discrete_params], [target.name], use_torch=True
)


def shifted_hartmann(x: Tensor, /) -> Tensor:
    """Calculate a shifted, scaled and noisy variant of the Hartmann function."""
    return 2.5 * Hartmann(dim=DIMENSION, noise_std=0.15)(x) + 3.25


test_functions = {
    "Hartmann": wrapper(Hartmann(dim=DIMENSION)),
    "Shifted": wrapper(shifted_hartmann),
}

### Generating Lookup Tables

# We generate a single lookup table containing the target values of both functions at
# the given parameter grid.
# Parts of one lookup serve as the training data for the model.
# The other lookup is used as the loop-closing element, providing the target values of
# the other function.

grid = np.meshgrid(*[p.values for p in discrete_params])

lookups: dict[str, pd.DataFrame] = {}
for function_name, function in test_functions.items():
    lookup = pd.DataFrame({f"x{d}": grid_d.ravel() for d, grid_d in enumerate(grid)})
    lookup = pd.concat([lookup, function(lookup)], axis=1)
    lookup["Function"] = function_name
    lookups[function_name] = lookup
lookup = pd.concat([lookups["Hartmann"], lookups["Shifted"]]).reset_index()

### Simulation Loop

campaign = Campaign(searchspace=searchspace, objective=objective)

results = simulate_transfer_learning(
    campaign,
    lookup,
    batch_size=BATCH_SIZE,
    n_doe_iterations=N_DOE_ITERATIONS,
    n_mc_iterations=N_MC_ITERATIONS,
)

# For comparison, we also compare with the baseline tasks

# ```{note}
# It is intended to implement a more elegant way of comparing results with and
# without transfer learning in the future.
# ```

for func_name, function in test_functions.items():
    task_param = TaskParameter(
        name="Function", values=["Hartmann", "Shifted"], active_values=[func_name]
    )
    parameters = [*discrete_params, task_param]
    searchspace = SearchSpace.from_product(parameters=parameters)
    result_baseline = simulate_scenarios(
        {f"{func_name}_No_TL": Campaign(searchspace=searchspace, objective=objective)},
        lookups[func_name],
        batch_size=BATCH_SIZE,
        n_doe_iterations=N_DOE_ITERATIONS,
        n_mc_iterations=N_MC_ITERATIONS,
    )

    results = pd.concat([results, result_baseline])

# All that remains is to visualize the results.
# As the example shows, the optimization speed can be significantly increased by
# using even small amounts of training data from related optimization tasks.

results.rename(columns={"Scenario": "Function"}, inplace=True)
# Add column to enable different styles for non-TL examples
results["Uses TL"] = results["Function"].apply(lambda val: "No_TL" not in val)
ax = sns.lineplot(
    data=results,
    markers=["o", "s"],
    markersize=13,
    x="Num_Experiments",
    y="Target_CumBest",
    hue="Function",
    style="Uses TL",
)
create_example_plots(ax=ax, base_name="backtesting")
