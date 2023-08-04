"""
Run history simulation for a single target with a custom defined function as lookup in
a hybrid searchspace.

This example shows a full simulation loop. That is, we do not only perform one or two
iterations like in the other examples but rather a full loop. We also store and display
the results. We refer to the examples in the searchspace folder for more information
on how to use the synthetic test functions and to the example file on hybrid
searchspaces for more information on them.

Note that this example assumes some familiarity with BayBE.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from baybe import BayBE
from baybe.parameters import NumericalContinuousParameter, NumericalDiscreteParameter
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.strategies import (
    NaiveHybridRecommender,
    RandomRecommender,
    SequentialGreedyRecommender,
    Strategy,
)
from baybe.targets import NumericalTarget, Objective

# For the full simulation, we need to define some additional parameters.
# These are the number of Monte Carlo runs and the number of experiments to be
# conducted per Monte Carlo run.
N_MC_ITERATIONS = 2
N_EXP_ITERATIONS = 5

# Here, we now need to define our custom function first. The function should accept an
# arbitrary or fixed amount of floats as input and return either a single float or
# a tuple of floats.
# NOTE It is assumed that the analytical test function does only perform a single
# calculation, i.e., it is assumed to work in a non-batched-way!


# We implement a simple sum of squares function with a single output.
def sum_of_squares(*x: float) -> float:
    """
    Calculates the sum of squares.
    """
    res = 0
    for y in x:
        res += y**2
    return res


# For our actual experiment, we need to specify the number of dimensions that we want
# to use as this is necessary to know for the creation of the parameters. The same is
# true for the bounds of the parameters which should be provided as a list of
# two-dimensional tuples.
DIMENSION = 4
BOUNDS = [
    (-2, 2),
    (-2, 2),
    (-2, 2),
    (-2, 2),
]

# Since we are in a hybrid setting, we need to specify which indices should be discrete
# and which should be continuous.
CONT_INDICES = [0, 1]
DISC_INDICES = [2, 3]

# This parameter decides how many points each discrete dimension should have
# NOTE This example uses the SequentialGreedyRecommender which performs a brute-force
# optimization over the discrete subspace. We thus heavily advise to keep the number
# of discrete parmeters and points rather small here.
POINTS_PER_DIM = 6

# Check if the provided indices match the given dimension
if set(CONT_INDICES + DISC_INDICES) != set(range(DIMENSION)):
    raise ValueError(
        "Either the intersection between CONT_IND and DISC_IND is not empty or your "
        "indices do not match."
    )

# Construct the continuous parameters
cont_parameters = [
    NumericalContinuousParameter(
        name=f"x_{k+1}",
        bounds=(BOUNDS[k]),
    )
    for k in CONT_INDICES
]

# Construct the discrete parameters
disc_parameters = [
    NumericalDiscreteParameter(
        name=f"x_{k+1}",
        values=list(np.linspace(*BOUNDS[k], POINTS_PER_DIM)),
        tolerance=0.01,
    )
    for k in DISC_INDICES
]

# Concatenate the continuous and discrete parameters
parameters = cont_parameters + disc_parameters

# Construct searchspace and objective.
searchspace = SearchSpace.from_product(parameters=parameters)

objective = Objective(
    mode="SINGLE", targets=[NumericalTarget(name="Target", mode="MIN")]
)

# This example compares the sequential greedy recommender with the naive hybrid
# recommender. We thus initialize a baybe object for both of them as well as for the
# random recommender.
seq_greedy_strategy = Strategy(
    recommender=SequentialGreedyRecommender(),
)
naive_hybrid_strategy = Strategy(recommender=NaiveHybridRecommender())
random_strategy = Strategy(recommender=RandomRecommender())


# Create the BayBE objects
seq_greedy_baybe = BayBE(
    searchspace=searchspace,
    strategy=seq_greedy_strategy,
    objective=objective,
)
naive_hybrid_baybe = BayBE(
    searchspace=searchspace,
    strategy=naive_hybrid_strategy,
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
# mapping names for the scenarios to actual BayBE objects
scenarios = {
    "Sequential greedy": seq_greedy_baybe,
    "Naive hybrid": naive_hybrid_baybe,
    "Random": random_baybe,
}
results = simulate_scenarios(
    scenarios=scenarios,
    batch_quantity=2,
    n_exp_iterations=N_EXP_ITERATIONS,
    n_mc_iterations=N_MC_ITERATIONS,
    lookup=sum_of_squares,
)

# The following lines plot the results and save the plot in run_analytical.png
sns.lineplot(data=results, x="Num_Experiments", y="Target_CumBest", hue="Variant")
plt.gcf().set_size_inches(24, 8)
plt.savefig("./run_hybrid.png")
