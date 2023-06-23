"""
Run history simulation for a single target with a custom defined function as lookup.

This example shows a full simulation loop. That is, we do not only perform one or two
iterations like in the other examples but rather a full loop. We also store and display
the results. We refer to the examples in the searchspace folder for more information
on how to use the synthetic test functions.

This example can also be used or testing new aspects like recommenders.

Note that this example assumes some familiarity with BayBE.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from baybe.core import BayBE

# Note that this import here might be problematic depending on your exact
# setup and that you might need to make some adjustments to make it work!
from baybe.parameters import NumericDiscrete
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.strategies.bayesian import SequentialGreedyRecommender
from baybe.strategies.sampling import RandomRecommender
from baybe.strategies.strategy import Strategy
from baybe.targets import NumericalTarget, Objective


# For the full simulation, we need to define some additional parameters.
# These are the number of Monte Carlo runs and the number of experiments to be
# conducted per Monte Carlo run.
N_MC_ITERATIONS = 2
N_EXP_ITERATIONS = 4

# Here, we now need to define our custom function first. The function should accept an
# arbitrary or fixed amount of floats as input and return either a single float or
# a tuple of floats.
# NOTE It is assumed that the analytical test function does only perform a single
# calculation, i.e., it is assumed to work in a non-batched-way!
# NOTE The case of using an analytical test function for multi-target optimization has
# not yet been fully tested.


# We implements a simple sum of squares function with a single output.
def sum_of_squares(*x: float) -> float:
    """
    Calculates the sum of squares.
    """
    res = 0
    for y in x:
        res += y**2
    return res


# For our actual experiment, we need to specify the number of dimension that we want
# to use as this is necessary to know for the creation of the parameters. The same is
# true for the bounds of the parameters which should be provided as a list of
# two-dimensional tuples.
DIMENSION = 4
BOUNDS = [(-2, 2), (-2, 2), (-2, 2), (-2, 2)]

# As we expect it to be the most common use case, we construct a purely discrete space
# here. We refer to the examples within examples/SearchSpaces for details on
# how to change this code for continuous or hybrid spaces.
parameters = [
    NumericDiscrete(
        name=f"x_{k+1}",
        values=list(np.linspace(*BOUNDS[k], 15)),
        tolerance=0.01,
    )
    for k in range(DIMENSION)
]

# Construct searchspace and objective.
searchspace = SearchSpace.create(parameters=parameters)

objective = Objective(
    mode="SINGLE", targets=[NumericalTarget(name="Target", mode="MIN")]
)

# The goal of this example is to demonstrate how to perform a full simulation. To
# simplify adjusting the example for other recommenders or strategies, we explicitly
# construct some strategy objects. For details on the construction of strategy objects,
# we refer to strategies.py within examples/Basics.

seq_greedy_EI_strategy = Strategy(
    recommender=SequentialGreedyRecommender(acquisition_function_cls="qEI"),
)
random_strategy = Strategy(recommender=RandomRecommender())


# Create the BayBE object
seq_greedy_EI_baybe = BayBE(
    searchspace=searchspace,
    strategy=seq_greedy_EI_strategy,
    objective=objective,
)
random_baybe = BayBE(
    searchspace=searchspace,
    strategy=random_strategy,
    objective=objective,
)

# We can now use the simulate_scenarios function from simulation.py to simulate a
# full experiment. Note that this function enables to run multiple scenarios one after
# another by a single function call, which is why we need to define a dictionary
# mapping names for the scenarios to actual baybe objects
scenarios = {
    "Sequential greedy EI": seq_greedy_EI_baybe,
    "Random": random_baybe,
}
results = simulate_scenarios(
    scenarios=scenarios,
    batch_quantity=3,
    n_exp_iterations=N_EXP_ITERATIONS,
    n_mc_iterations=N_MC_ITERATIONS,
    lookup=sum_of_squares,
)

# The following lines plot the results and save the plot in run_analytical.png
sns.lineplot(data=results, x="Num_Experiments", y="Target_CumBest", hue="Variant")
plt.gcf().set_size_inches(24, 8)
plt.savefig("./run_analytical.png")
