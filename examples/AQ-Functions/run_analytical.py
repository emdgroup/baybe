"""
Test to validate if the initial strategies work as expected
"""
# pylint: disable=W0611
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from baybe.core import BayBE
from baybe.parameters import NumericContinuous, NumericDiscrete
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.strategies.bayesian import SequentialGreedyRecommender
from baybe.strategies.sampling import RandomRecommender

from baybe.strategies.strategy import Strategy
from baybe.targets import NumericalTarget, Objective

from examples.Analytic_Functions.test_functions import Hartmann3, Rastrigin  # noqa

# --------------------------------------------------------------------------------------
# PART 1: Definition of test case
# --------------------------------------------------------------------------------------
# Choice of test case here. Test function is simply the imported function. Note that it
# is necessary to specify the DIMENSIONS only for those test functions that do not
# already specify their test function in their name.

# Part 1.1: Details for the creation of BayBE objects
# Note that the non-capitalised part will be varied in this test example, so do not
# change it here.
INITIAL_RECOMMENDER = RandomRecommender()
ALLOW_REPEATED_RECOMMENDATIONS = False
ALLOW_RECOMMENDING_ALREADY_MEASURED = False
NUM_MEAS_WITHIN_TOLERANCE = True
recommender = None  # pylint: disable=C0103

# Part 1.2: Experimental details for running the simulation
# Change as you see fit!
# NOTE This is currently configured for discrete or continuous subspaces only.
DIMENSIONS = 3
TEST_FUNCTION = Rastrigin(dimension=DIMENSIONS)
CONT_IND = []
DISC_IND = [0, 1, 2]
POINTS_PER_DIM = 25
N_EXP_ITERATIONS = 10
N_MC_ITERATIONS = 5
BATCH_QUANTITY = 3

# Part 1.3: Defining a name for the test case
file_name = (
    f"{TEST_FUNCTION}_IT{N_EXP_ITERATIONS}_MC{N_MC_ITERATIONS}_BQ{BATCH_QUANTITY}"
)

# --------------------------------------------------------------------------------------
# PART 2: Checking the user inputs
# --------------------------------------------------------------------------------------
# This part performs some very minor tests to see whether the inputs given in Part 1
# make sense.

if set(CONT_IND + DISC_IND) != set(range(TEST_FUNCTION.dimension)):
    raise ValueError(
        "Either the tntersection between CONT_IND and DISC_IND is not empty or\
              your indices do not match."
    )
if len(CONT_IND) != 0 and len(DISC_IND) != 0:
    raise NotImplementedError("This example is not yet available for hybrid spaces.")

# --------------------------------------------------------------------------------------
# PART 3: Creating the BayBE objects
# --------------------------------------------------------------------------------------
# This part collects the user inputs and creates the corresponding BayBE objects.

disc_parameters = [
    NumericDiscrete(
        name=f"x_{k+1}",
        values=list(np.linspace(*TEST_FUNCTION.bounds[k], POINTS_PER_DIM)),
        tolerance=0.01,
    )
    for k in DISC_IND
]

cont_parameters = [
    NumericContinuous(name=f"x_{k+1}", bounds=TEST_FUNCTION.bounds[k]) for k in CONT_IND
]

searchspace = SearchSpace.create(
    parameters=disc_parameters + cont_parameters,
)

objective = Objective(
    mode="SINGLE", targets=[NumericalTarget(name="Target", mode="MIN")]
)

acq_functions = (
    ["qPI", "qEI", "qUCB"]
    if BATCH_QUANTITY > 1
    else ["qPI", "qEI", "qUCB", "PM", "PI", "EI", "UCB"]
)
stratgy_dict = {}
for acq_function in acq_functions:
    stratgy_dict[acq_function] = Strategy(
        initial_recommender=INITIAL_RECOMMENDER,
        recommender=SequentialGreedyRecommender(acquisition_function_cls=acq_function),
        allow_repeated_recommendations=ALLOW_REPEATED_RECOMMENDATIONS,
        allow_recommending_already_measured=ALLOW_RECOMMENDING_ALREADY_MEASURED,
    )

# ---------------------------------------------------------------------------------------
# PART 4: PUTTING THINGS TOGETHER
# ---------------------------------------------------------------------------------------
# For putting things together, we define all test scenarios and simulate them.
# Afterwards, the results are plotted and saved in a file.

scenarios = {
    name: BayBE(
        searchspace=searchspace,
        strategy=strategy,
        objective=objective,
        numerical_measurements_must_be_within_tolerance=NUM_MEAS_WITHIN_TOLERANCE,
    )
    for name, strategy in stratgy_dict.items()
}

results = simulate_scenarios(
    scenarios=scenarios,
    lookup=TEST_FUNCTION,  # pylint: disable=E0601
    n_exp_iterations=N_EXP_ITERATIONS,
    n_mc_iterations=N_MC_ITERATIONS,
    batch_quantity=BATCH_QUANTITY,
)

print(results)

fig = sns.lineplot(data=results, x="Num_Experiments", y="Target_CumBest", hue="Variant")
fig.axhline(TEST_FUNCTION.optimum, label="Optimum", ls="--", c="black")
plt.gcf().set_size_inches(24, 8)
plt.legend()
plt.savefig(f"./{file_name}.png")
