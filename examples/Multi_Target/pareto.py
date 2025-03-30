## Single-Target vs. Pareto Optimization

# In this example, we illustrate the difference between single-target and Pareto
# optimization under laboratory conditions provided by a pair of synthetic targets.
# In particular:
# * We set up two quadratic target functions with different center points,
# * visualize the corresponding Pareto frontier,
# * and compare the recommendations obtained by optimizing the targets individually
#   and jointly.


### Imports

import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from baybe.campaign import Campaign
from baybe.objectives import ParetoObjective
from baybe.parameters import NumericalContinuousParameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from baybe.utils.dataframe import arrays_to_dataframes
from baybe.utils.random import set_random_seed

### Settings

# Let's first define some general settings for our example:

SMOKE_TEST = "SMOKE_TEST" in os.environ

BATCH_SIZE = 2 if SMOKE_TEST else 10
N_TRAINING_DATA = 2 if SMOKE_TEST else 100
N_GRID_POINTS = 3 if SMOKE_TEST else 100
CENTER_Y0 = (-0.5, -0.5)
CENTER_Y1 = (0.5, 0.5)

# Also, we fix the random seed for reproducibility:

set_random_seed(42)


### Defining the Optimization Problem

# We start by defining the parameters and targets describing the inputs and outputs
# of our synthetic functions:

x0 = NumericalContinuousParameter("x0", (-1, 1))
x1 = NumericalContinuousParameter("x1", (-1, 1))
y0 = NumericalTarget("y0", "MAX")
y1 = NumericalTarget("y1", "MIN")
searchspace = SearchSpace.from_product([x0, x1])

# With these definitions at hand, we can construct a multi-variate callable representing
# the two quadratic target functions:


@arrays_to_dataframes([x0.name, x1.name], [y0.name, y1.name])
def lookup(arr: np.ndarray) -> np.ndarray:
    """Compute (negative scaled) mean square values for different center points."""
    y0 = -np.sum((arr - CENTER_Y0) ** 2, axis=1)  # negative
    y1 = np.sum((arr - CENTER_Y1) ** 2, axis=1) * 100 + 1000  # scaled
    return np.c_[y0, y1]


### Campaign Setup

# We now query the callable with some randomly generated inputs to collect training
# data for our model:

data = searchspace.continuous.sample_uniform(N_TRAINING_DATA)
data = pd.concat([data, lookup(data)], axis=1)

# Next, we create three campaigns for comparison:
# * One focusing on the first target
# * One focusing on the second target
# * One for Pareto optimization of both targets

campaign_y0 = Campaign(searchspace=searchspace, objective=y0.to_objective())
campaign_y1 = Campaign(searchspace=searchspace, objective=y1.to_objective())
campaign_par = Campaign(searchspace=searchspace, objective=ParetoObjective([y0, y1]))

# We feed each campaign with the same training data and request recommendations:

campaign_y0.add_measurements(data)
campaign_y1.add_measurements(data)
campaign_par.add_measurements(data)

rec_y0 = campaign_y0.recommend(BATCH_SIZE)
rec_y1 = campaign_y1.recommend(BATCH_SIZE)
rec_par = campaign_par.recommend(BATCH_SIZE)

out_y0 = lookup(rec_y0)
out_y1 = lookup(rec_y1)
out_par = lookup(rec_par)


### Visualization

# To visualize the results, we first create grids to sample our target functions:

x0_mesh, x1_mesh = np.meshgrid(
    np.linspace(*x0.bounds.to_tuple(), N_GRID_POINTS),
    np.linspace(*x1.bounds.to_tuple(), N_GRID_POINTS),
)
df_y = lookup(pd.DataFrame({x0.name: x0_mesh.ravel(), x1.name: x1_mesh.ravel()}))
y0_mesh = np.reshape(df_y[y0.name], x0_mesh.shape)
y1_mesh = np.reshape(df_y[y1.name], x1_mesh.shape)


# Now, we can plot the function values, the training data, the recommendations,
# and the Pareto frontier in the parameter space:

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

plt.sca(axs[0])
plt.contour(x0_mesh, x1_mesh, y0_mesh, colors="tab:red", alpha=0.2)
plt.contour(x0_mesh, x1_mesh, y1_mesh, colors="tab:blue", alpha=0.2)
plt.plot(*np.c_[CENTER_Y0, CENTER_Y1], "k", label="frontier")
plt.plot(data[x0.name], data[x1.name], "o", color="0.7", markersize=2, label="training")
plt.plot(rec_y0[x0.name], rec_y0[x1.name], "o", color="tab:red", label="single_y0")
plt.plot(rec_y1[x0.name], rec_y1[x1.name], "o", color="tab:blue", label="single_y1")
plt.plot(rec_par[x0.name], rec_par[x1.name], "o", color="tab:purple", label="pareto")
plt.legend(loc="upper left")
plt.xlabel(x0.name)
plt.ylabel(x1.name)
plt.title("Parameter Space")
plt.axis("square")
plt.axis([-1, 1, -1, 1])

# Similarly, we plot the training data, the achieved function values,
# and the Pareto frontier in the target space:

plt.sca(axs[1])
frontier = lookup(
    pd.DataFrame(np.linspace(CENTER_Y0, CENTER_Y1), columns=[x0.name, x1.name])
)
plt.plot(*frontier.to_numpy().T, "k", label="frontier")
plt.plot(data[y0.name], data[y1.name], "o", color="0.7", markersize=2, label="training")
plt.plot(out_y0[y0.name], out_y0[y1.name], "o", color="tab:red", label="single_y0")
plt.plot(out_y1[y0.name], out_y1[y1.name], "o", color="tab:blue", label="single_y1")
plt.plot(out_par[y0.name], out_par[y1.name], "o", color="tab:purple", label="pareto")
plt.legend(loc="upper left")
plt.xlabel(y0.name)
plt.ylabel(y1.name)
plt.title("Target Space")

plt.tight_layout()
if not SMOKE_TEST:
    plt.savefig("pareto.svg")
