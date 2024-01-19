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
from botorch.test_functions import SixHumpCamel, ThreeHumpCamel

from baybe import Campaign
from baybe.objective import Objective
from baybe.parameters import NumericalDiscreteParameter, TaskParameter
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios, simulate_transfer_learning
from baybe.targets import NumericalTarget
from baybe.utils import botorch_function_wrapper

#### Parameters for a full simulation loop

# For the full simulation, we need to define some additional parameters.
# These are the number of Monte Carlo runs and the number of experiments to be conducted per run.

N_MC_ITERATIONS = 9
N_DOE_ITERATIONS = 5

#### Defining the tasks

# We use two similar two-dimensional test functions imported from BoTorch.
# We use them to create a lookup table after creating the search space and the objective.
three_hump_camel = botorch_function_wrapper(ThreeHumpCamel())
six_hump_camel = botorch_function_wrapper(SixHumpCamel())

#### Creating the searchspace and the objective

# The parameter `POINTS_PER_DIM` controls the number of points per dimension.
# Note that the searchspace will have `POINTS_PER_DIM**2` many points.

POINTS_PER_DIM = 10

# We define two numerical discrete parameters, as well as a ``TaskParameter``.
# This parameter contains the information about the different tasks that we have in this
# transfer learning example.
x1 = NumericalDiscreteParameter(
    name="x_1", values=list(np.linspace(-2, 2, POINTS_PER_DIM))
)
x2 = NumericalDiscreteParameter(
    name="x_2", values=list(np.linspace(-2, 2, POINTS_PER_DIM))
)
task_param = TaskParameter(name="Function", values=("ThreeHump", "SixHump"))
parameters = [x1, x2, task_param]

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

x = x1.values
y = x2.values
xx, yy = np.meshgrid(x, y)


lookup_3hc = pd.DataFrame({"x_1": xx.flatten(), "x_2": yy.flatten()})
lookup_3hc["Target"] = lookup_3hc.apply(three_hump_camel, axis=1)
lookup_3hc["Function"] = "ThreeHump"

lookup_6hc = pd.DataFrame({"x_1": xx.flatten(), "x_2": yy.flatten()})
lookup_6hc["Target"] = lookup_6hc.apply(six_hump_camel, axis=1)
lookup_6hc["Function"] = "SixHump"

lookup = pd.concat([lookup_3hc, lookup_6hc], ignore_index=True)

# We print the first ten rows to show how the table looks like
print(lookup.head(10))

#### Performing the simulation loop

# We can now use the `simulate_transfer_learning` function to simulate a full experiment.

results_with_tf = simulate_transfer_learning(
    campaign,
    lookup,
    batch_quantity=1,
    n_doe_iterations=N_DOE_ITERATIONS,
    n_mc_iterations=N_MC_ITERATIONS,
)

# To showcase the improvement, we also optimize the same tasks independently and use
# the `simulate_scenarios` function.

parameters_without_tl = [x1, x2]
campaign_without_tl_3hc = Campaign(
    searchspace=SearchSpace.from_product([x1, x2]), objective=objective
)
campaign_without_tl_6hc = Campaign(
    searchspace=SearchSpace.from_product([x1, x2]), objective=objective
)

scenarios_3hc = {"ThreeHump no TL": campaign_without_tl_3hc}
scenarios_6hc = {"SixHump no TL": campaign_without_tl_6hc}

results_3hc_no_tl = simulate_scenarios(
    scenarios_3hc,
    three_hump_camel,
    batch_quantity=1,
    n_doe_iterations=N_DOE_ITERATIONS,
    n_mc_iterations=N_MC_ITERATIONS,
)

results_6hc_no_tl = simulate_scenarios(
    scenarios_6hc,
    six_hump_camel,
    batch_quantity=1,
    n_doe_iterations=N_DOE_ITERATIONS,
    n_mc_iterations=N_MC_ITERATIONS,
)


# We concatenate all the results and show them in a single plot.
results = pd.concat([results_with_tf, results_3hc_no_tl, results_6hc_no_tl])
sns.lineplot(data=results, x="Num_Experiments", y="Target_CumBest", hue="Scenario")
plt.gcf().set_size_inches(24, 8)
plt.savefig("./run_analytical.png")
