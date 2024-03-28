## Transfer Learning

# This example demonstrates BayBE's
# {doc}`Transfer Learning </userguide/transfer_learning>` capabilities using the
# Hartmann test function:
# * We construct a campaign,
# * give it access to data from a related but different task,
# * and show how this additional information boosts optimization performance.

### Imports

import os
import sys
from pathlib import Path

import pandas as pd
import seaborn as sns
from botorch.test_functions.synthetic import Hartmann

from baybe import Campaign
from baybe.objective import Objective
from baybe.parameters import NumericalContinuousParameter, TaskParameter
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget
from baybe.utils.botorch_wrapper import botorch_function_wrapper
from baybe.utils.plotting import create_example_plots

### Settings

# The following settings are used to set up the problem:

SMOKE_TEST = "SMOKE_TEST" in os.environ  # reduce the problem complexity in CI pipelines
DIMENSION = 3  # input dimensionality of the test function
BATCH_SIZE = 1  # batch size of recommendations per DOE iteration
N_MC_ITERATIONS = 2 if SMOKE_TEST else 5  # number of Monte Carlo runs
N_DOE_ITERATIONS = 2 if SMOKE_TEST else 20  # number of DOE iterations
POINTS_PER_DIM = 3 if SMOKE_TEST else 5  # number of grid points per input dimension


### Creating the Optimization Objective

# The test functions each have a single output that is to be minimized.
# The corresponding [Objective](baybe.objective.Objective)
# is created as follows:

objective = Objective(
    mode="SINGLE", targets=[NumericalTarget(name="Target", mode="MIN")]
)

### Creating the Searchspace

# The bounds of the search space are dictated by the test function:

BOUNDS = Hartmann(dim=DIMENSION).bounds

# First, we define one
# [NumericalDiscreteParameter](baybe.parameters.numerical.NumericalDiscreteParameter)
# per input dimension of the test function:

params = [
    NumericalContinuousParameter(
        name=f"x{d}",
        bounds=(lower, upper),
    )
    for d, (lower, upper) in enumerate(BOUNDS.T)
]

# ```{note}
# While we could optimize the function using
# [NumericalContinuousParameters](baybe.parameters.numerical.NumericalContinuousParameter),
# we use discrete parameters here because it lets us interpret the percentages shown in
# the final plot directly as the proportion of candidates for which there were target
# values revealed by the training function.
# ```

# Next, we define a
# [TaskParameter](baybe.parameters.categorical.TaskParameter) to encode the task context,
# which allows the model to establish a relationship between the training data and
# the data collected during the optimization process.
# Because we want to obtain recommendations only for the test function, we explicitly
# pass the `active_values` keyword.

task_param = TaskParameter(
    name="Function",
    values=["Test_Function", "Training_Function"],
    active_values=["Test_Function"],
)

# With the parameters at hand, we can now create our search space.

parameters = [*params, task_param]
searchspace = SearchSpace.from_product(parameters=parameters)

### Defining the Tasks

# To demonstrate the transfer learning mechanism, we consider the problem of optimizing
# the Hartmann function using training data from its negated version, including some
# noise. The used model is of course not aware of this relationship but needs to infer
# it from the data gathered during the optimization process.


def shifted_hartmann(*x: float) -> float:
    """Calculate a shifted, scaled and noisy variant of the Hartman function."""
    noised_hartmann = Hartmann(dim=DIMENSION)
    return 2.5 * botorch_function_wrapper(noised_hartmann)(x) + 3.25


test_functions = {
    "Test_Function": botorch_function_wrapper(Hartmann(dim=DIMENSION)),
    "Training_Function": shifted_hartmann,
}

# (Lookup)=
### Generating Lookup Tables

# We generate two lookup tables containing the target values of both test
# functions at the given parameter grid.
# Parts of one lookup serve as the training data for the model.
# The other lookup is used as the loop-closing element, providing the target values of
# the test functions on demand.


### Simulation Loop

# We now simulate campaigns for different amounts of training data unveiled,
# to show the impact of transfer learning on the optimization performance.
# To average out and reduce statistical effects that might happen due to the random
# sampling of the provided data, we perform several Monte Carlo runs.

results: list[pd.DataFrame] = []
for n in (10, 100, 1000):
    initial_data = []
    for _ in range(N_MC_ITERATIONS):
        data = searchspace.continuous.samples_random(n_points=n)
        data["Target"] = data.apply(test_functions["Training_Function"], axis=1)
        data["Function"] = "Training_Function"
        initial_data.append(data)

    campaign = Campaign(searchspace=searchspace, objective=objective)
    result_fraction = simulate_scenarios(
        {f"{n}": campaign},
        test_functions["Test_Function"],
        initial_data=initial_data,
        batch_size=BATCH_SIZE,
        n_doe_iterations=N_DOE_ITERATIONS,
    )
    results.append(result_fraction)

results = pd.concat(*results)

# All that remains is to visualize the results.
# As the example shows, the optimization speed can be significantly increased by
# using even small amounts of training data from related optimization tasks.

results.rename(columns={"Scenario": "% of data used"}, inplace=True)
path = Path(sys.path[0])
ax = sns.lineplot(
    data=results,
    marker="o",
    markersize=10,
    x="Num_Experiments",
    y="Target_CumBest",
    hue="% of data used",
)
create_example_plots(
    ax=ax,
    path=path,
    base_name="basic_transfer_learning",
)
