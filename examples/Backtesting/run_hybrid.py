### Example for full simulation loop using a custom analytical test function in a hybrid space

"""
This example shows a simulation loop for a single target with a custom test function as lookup.
Most importantly, it demonstrates the creation of a custom hybrid searchspace.
"""

# This examples assumes some basic familiarity with using BayBE and the lookup mechanism.
# We refer to [`baybe_object`](./../Basics/baybe_object.md) for a more  basic example resp.
# to [`run_custom_analytical`](./run_custom_analytical.md) for details on the lookup mechanism.

#### Necessary imports for this example

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


#### Parameters for a full simulation loop

# For the full simulation, we need to define some additional parameters.
# These are the number of Monte Carlo runs and the number of experiments to be conducted per run.
N_MC_ITERATIONS = 2
N_EXP_ITERATIONS = 2

#### Defining the test function.


# See [`here`](./run_custom_analytical.md) for details on the custom analytical test function.
def sum_of_squares(*x: float) -> float:
    """
    Calculates the sum of squares.
    """
    res = 0
    for y in x:
        res += y**2
    return res


# For our actual experiment, we need to specify the number of dimension that we want to use.
# This is necessary to know for the creation of the parameters.
# Similarly, it is necessary to state the bounds of the parameters.
# These should be provided as a list of two-dimensional tuples.
DIMENSION = 4
BOUNDS = [(-2, 2), (-2, 2), (-2, 2), (-2, 2)]

#### Constructing the hybrid searchspace

# Our goal is to construct a hybrid searchspace containing discrete and continuous parameters.
# We thus need to specify which indices should be discrete and which should be continuous.
CONT_INDICES = [0, 1]
DISC_INDICES = [2, 3]

# This code verifies whether the provided indices agree with `DIMENSION`.
if set(CONT_INDICES + DISC_INDICES) != set(range(DIMENSION)):
    raise ValueError(
        "Either the intersection between CONT_IND and DISC_IND is not empty or your "
        "indices do not match."
    )

# The following parameter decides how many points each discrete dimension should have.
# Note that this example uses the `SequentialGreedyRecommender` (among others).
# This recommender performs a brute-force optimization over the discrete subspace.
# We thus heavily advise to keep the number of discrete parmeters and points rather small here.
POINTS_PER_DIM = 6


# Construct the continuous parameters as NumericContinuous parameters.
cont_parameters = [
    NumericalContinuousParameter(
        name=f"x_{k+1}",
        bounds=(BOUNDS[k]),
    )
    for k in CONT_INDICES
]

# Construct the discrete parameters as `NumericalDiscreteParameters`.
disc_parameters = [
    NumericalDiscreteParameter(
        name=f"x_{k+1}",
        values=list(np.linspace(*BOUNDS[k], POINTS_PER_DIM)),
        tolerance=0.01,
    )
    for k in DISC_INDICES
]

# Concatenate the continuous and discrete parameters.
parameters = cont_parameters + disc_parameters

# Construct searchspace and objective.
searchspace = SearchSpace.from_product(parameters=parameters)
objective = Objective(
    mode="SINGLE", targets=[NumericalTarget(name="Target", mode="MIN")]
)

#### Constructing BayBE objects for the simulation loop

# This example compares three different available hybrid recommenders:
# The `SequentialGreedyRecommender`, the `NaiveHybridRecommedner` and the `RandomRecommender`.
# For each of them, we initialize one strategy object.
# Note that it is possible to further specify the behavor of the `SequentialGreedyRecommender`.
# Using the two keywords `hybrid_sampler` and `sampling_percentage`, one can control
# - how much of the discrete subspace should be explored
# - how these points should be sampled.

# Note that the recommender performs one optimization of the continuous subspace per sampled point.
# We thus recommend to keep this parameter rather low.

seq_greedy_strategy = Strategy(
    recommender=SequentialGreedyRecommender(
        hybrid_sampler="Farthest", sampling_percentage=0.3
    ),
)
naive_hybrid_strategy = Strategy(recommender=NaiveHybridRecommender())
random_strategy = Strategy(recommender=RandomRecommender())

# We now create one BayBE object per strategy.

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

# We can now use the `simulate_scenarios` function to simulate a full experiment.
# Note that this function enables to run multiple scenarios by a single function call.
# For this, it is necessary to define a dictionary mapping scenario names to BayBE objects.
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
