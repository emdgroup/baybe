### Example for full simulation loop using a BoTorch test function

# This example shows a simulation loop for a single target with a BoTorch test function as lookup.
# That is, we perform several Monte Carlo runs with several iterations.
# In addition, we also store and display the results.

# This example assumes some basic familiarity with using BayBE and how to use BoTorch test
# functions in discrete searchspaces.
# We thus refer to
# 1. [`campaign`](./../Basics/campaign.md) for a basic example on how to use BayBE and
# 2. [`discrete_space`](./../Searchspaces/discrete_space.md) for details on using a
# BoTorch test function.

#### Necessary imports for this example

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from botorch.test_functions import Rastrigin

from baybe import Campaign
from baybe.objective import Objective
from baybe.parameters import NumericalDiscreteParameter
from baybe.recommenders import RandomRecommender, SequentialGreedyRecommender
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.strategies import TwoPhaseStrategy
from baybe.targets import NumericalTarget
from baybe.utils import botorch_function_wrapper

#### Parameters for a full simulation loop

# For the full simulation, we need to define some additional parameters.
# These are the number of Monte Carlo runs and the number of experiments to be conducted per run.

N_MC_ITERATIONS = 2
N_DOE_ITERATIONS = 2

#### Defining the test function

# See [`discrete_space`](./../Searchspaces/discrete_space.md) for details.

DIMENSION = 4
TestFunctionClass = Rastrigin

if not hasattr(TestFunctionClass, "dim"):
    TestFunction = TestFunctionClass(dim=DIMENSION)
else:
    print(
        f"\nYou choose a dimension of {DIMENSION} for the test function"
        f"{TestFunctionClass}. However, this function can only be used in "
        f"{TestFunctionClass().dim} dimension, so the provided dimension is replaced."
    )
    TestFunction = TestFunctionClass()
    DIMENSION = TestFunctionClass().dim

BOUNDS = TestFunction.bounds
WRAPPED_FUNCTION = botorch_function_wrapper(test_function=TestFunction)

#### Creating the searchspace and the objective

# The parameter `POINTS_PER_DIM` controls the number of points per dimension.
# Note that the searchspace will have `POINTS_PER_DIM**DIMENSION` many points.

POINTS_PER_DIM = 10
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

searchspace = SearchSpace.from_product(parameters=parameters)
objective = Objective(
    mode="SINGLE", targets=[NumericalTarget(name="Target", mode="MIN")]
)

#### Constructing campaigns for the simulation loop

# To simplify adjusting the example for other strategies, we construct some strategy objects.
# For details on strategy objects, we refer to [`strategies`](./../Basics/strategies.md).

seq_greedy_EI_strategy = TwoPhaseStrategy(
    recommender=SequentialGreedyRecommender(acquisition_function_cls="qEI"),
)
random_strategy = TwoPhaseStrategy(recommender=RandomRecommender())

# We now create one campaign per strategy.

seq_greedy_EI_campaign = Campaign(
    searchspace=searchspace,
    strategy=seq_greedy_EI_strategy,
    objective=objective,
)
random_campaign = Campaign(
    searchspace=searchspace,
    strategy=random_strategy,
    objective=objective,
)

#### Performing the simulation loop

# We can now use the `simulate_scenarios` function to simulate a full experiment.
# Note that this function enables to run multiple scenarios by a single function call.
# For this, it is necessary to define a dictionary mapping scenario names to campaigns.

scenarios = {
    "Sequential greedy EI": seq_greedy_EI_campaign,
    "Random": random_campaign,
}
results = simulate_scenarios(
    scenarios,
    WRAPPED_FUNCTION,
    batch_quantity=3,
    n_doe_iterations=N_DOE_ITERATIONS,
    n_mc_iterations=N_MC_ITERATIONS,
)

# The following lines plot the results and save the plot in run_analytical.png
sns.lineplot(data=results, x="Num_Experiments", y="Target_CumBest", hue="Scenario")
plt.gcf().set_size_inches(24, 8)
plt.savefig("./run_analytical.png")
