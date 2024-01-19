### Example for a simulation loop using the transfer learning capabilities

# This example shows how to use BayBE's transfer learning capabilities.
# That is, we demonstrate how to utilize information transfer between tasks.

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
from torch import Tensor

from baybe import Campaign
from baybe.objective import Objective
from baybe.parameters import NumericalDiscreteParameter, TaskParameter
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_transfer_learning
from baybe.targets import NumericalTarget

#### Parameters for a full simulation loop

# For the full simulation, we need to define some additional parameters.
# These are the number of Monte Carlo runs and the number of experiments to be conducted per run.

N_MC_ITERATIONS = 2
N_DOE_ITERATIONS = 4

#### Defining the tasks

# We use two similar two-dimensional test functions imported from BoTorch.
# We use them to create a lookup table after creating the search space and the objective.
three_hump_camel = ThreeHumpCamel()
six_hump_camel = SixHumpCamel()

#### Creating the searchspace and the objective

# The parameter `POINTS_PER_DIM` controls the number of points per dimension.
# Note that the searchspace will have `POINTS_PER_DIM**2` many points.

BOUNDS = three_hump_camel.bounds
POINTS_PER_DIM = 20

x1 = NumericalDiscreteParameter(
    name="x_1", values=list(np.linspace(BOUNDS[0, 0], BOUNDS[1, 0], POINTS_PER_DIM))
)
x2 = NumericalDiscreteParameter(
    name="x_2", values=list(np.linspace(BOUNDS[0, 1], BOUNDS[1, 1], POINTS_PER_DIM))
)

# We now append a ``TaskParameter``. This parameter contains the information about the
# different tasks that we have in this transfer learning example.
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
# parameter, including one for the TaskParameter. The values in that column describe
# for which task a measurement was made.

# Generate the grid coordinates using numpy
x = parameters[0].values
y = parameters[1].values
xx, yy = np.meshgrid(x, y)

lookup = pd.DataFrame({"x_1": xx.flatten(), "x_2": yy.flatten()})

# We randomly decide for each row which function we measure
lookup["Function"] = np.random.choice(["ThreeHump", "SixHump"], lookup.shape[0])


# This is a little helper function that is used to apply the function three_hump_camel
# resp. six_humP-camel if the value of the "Function" column is ``"ThreeHump"`` resp.
# ``"SixHump"``.
def _choose_task(x1, x2, function):
    func_dict = {"ThreeHump": three_hump_camel, "SixHump": six_hump_camel}
    return float(func_dict[function](Tensor((x1, x2))))


# Apply the helper function to the lookup dataframe
lookup["Target"] = lookup.apply(
    lambda x: _choose_task(x["x_1"], x["x_2"], x["Function"]), axis=1
)

# We print the first ten rows to show how the table looks like
print(lookup.head(10))


#### Performing the simulation loop

# We can now use the `simulate_transfer_learning` function to simulate a full experiment.

results = simulate_transfer_learning(
    campaign,
    lookup,
    batch_quantity=1,
    n_doe_iterations=N_DOE_ITERATIONS,
    n_mc_iterations=N_MC_ITERATIONS,
)

# The following lines plot the results and save the plot in run_analytical.png
sns.lineplot(data=results, x="Num_Experiments", y="Target_CumBest", hue="Scenario")
plt.gcf().set_size_inches(24, 8)
plt.savefig("./run_analytical.png")
