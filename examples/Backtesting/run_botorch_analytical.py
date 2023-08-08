"""
Run history simulation for a single target with a Botorch test function as lookup.

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

from baybe import BayBE
from baybe.parameters import NumericalDiscreteParameter
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.strategies import RandomRecommender, SequentialGreedyRecommender, Strategy
from baybe.targets import NumericalTarget, Objective
from baybe.utils import botorch_function_wrapper
from botorch.test_functions import Rastrigin


# For the full simulation, we need to define some additional parameters.
# These are the number of Monte Carlo runs and the number of experiments to be
# conducted per Monte Carlo run.
N_MC_ITERATIONS = 2
N_EXP_ITERATIONS = 5

# Here, you can choose the dimension  and the actual the test function.
# All BoTorch test functions can be used.
# Note that some test functions are only defined for specific dimensions.
# If the dimension you provide is not available for the given test function, a warning
# will be printed and the dimension is adjusted.
# For details on constructing the BayBE object, we refer to the basic example file.
DIMENSION = 4
TestFunctionClass = Rastrigin

# This part checks if the test function already has a fixed dimension.
# In that case, we print a warning and replace DIMENSION.
if not hasattr(TestFunctionClass, "dim"):
    TestFunction = TestFunctionClass(dim=DIMENSION)  # pylint: disable = E1123
else:
    print(
        f"\nYou choose a dimension of {DIMENSION} for the test function"
        f"{TestFunctionClass}. However, this function can only be used in "
        f"{TestFunctionClass().dim} dimension, so the provided dimension is replaced."
    )
    TestFunction = TestFunctionClass()
    DIMENSION = TestFunctionClass().dim

# Get the bounds of the variables as they are set by BoTorch
BOUNDS = TestFunction.bounds
# Create the wrapped function itself.
WRAPPED_FUNCTION = botorch_function_wrapper(test_function=TestFunction)
POINTS_PER_DIM = 15


# As we expect it to be the most common use case, we construct a purely discrete space
# here. We refer to the examples within examples/SearchSpaces for details on
# how to change this code for continuous or hybrid spaces.
parameters = [
    NumericalDiscreteParameter(
        name=f"x_{k+1}",
        values=list(
            np.linspace(
                BOUNDS[0, k],
                BOUNDS[1, k],
                POINTS_PER_DIM,
            )
        ),
        tolerance=0.01,
    )
    for k in range(DIMENSION)
]

# Construct searchspace and objective.
searchspace = SearchSpace.from_product(parameters=parameters)

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
# mapping names for the scenarios to actual BayBE objects
scenarios = {
    "Sequential greedy EI": seq_greedy_EI_baybe,
    "Random": random_baybe,
}
results = simulate_scenarios(
    scenarios=scenarios,
    batch_quantity=3,
    n_exp_iterations=N_EXP_ITERATIONS,
    n_mc_iterations=N_MC_ITERATIONS,
    lookup=WRAPPED_FUNCTION,
)

# The following lines plot the results and save the plot in run_analytical.png
sns.lineplot(data=results, x="Num_Experiments", y="Target_CumBest", hue="Variant")
plt.gcf().set_size_inches(24, 8)
plt.savefig("./run_analytical.png")
