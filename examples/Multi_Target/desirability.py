"""
This example shows how one can use a multi target mode for the objective.
It uses a desirability value to handle several targets.
It assumes that the reader is familiar with the basics of BayBE, and thus does not
explain the details of e.g. parameter creation. For additional explanation on these
aspects, we refer to the Basic examples.
"""

from baybe import BayBE
from baybe.parameters import CategoricalParameter, NumericalDiscreteParameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget, Objective
from baybe.utils import add_fake_results

# SEARCHSPACE
# --------------------------------------------------------------------------------------


# We begin by setting up some parameters for our experiments

Categorical_1 = CategoricalParameter("Categorical_1", values=[22, 33], encoding="OHE")
Categorical_2 = CategoricalParameter(
    "Categorical_2",
    values=["very bad", "bad", "OK", "good", "very good"],
    encoding="INT",
)
Num_disc_1 = NumericalDiscreteParameter(
    "Num_disc_1", values=[1, 2, 3, 4, 6, 8, 10], tolerance=0.3
)
Num_disc_2 = NumericalDiscreteParameter(
    "Num_disc_2", values=[-1, -3, -6, -9], tolerance=0.3
)

parameters = [Categorical_1, Categorical_2, Num_disc_1, Num_disc_2]

searchspace = SearchSpace.from_product(parameters=parameters)


# TARGETS
# --------------------------------------------------------------------------------------


# The multi target mode is handled when creating the objective object
# Thus we first need to define the different targets

# For example, we can start by defining two targets
# One which is maximized and one minimized during the optimization process

# Note that in this multi target mode,
# the user must specify bounds for each target

Target_1 = NumericalTarget(name="Target_1", mode="MAX", bounds=(0, 100))
Target_2 = NumericalTarget(name="Target_2", mode="MIN", bounds=(0, 100))


# For each target it is also possible to specify a Bounds_transformation_function
# This function is used to transform target values to an interval [0;1]
# for MAX and MIN mode, an ascending or decreasing 'LINEAR' function is used per default
# for match mode two functions are available 'TRIANGULAR' or 'BELL'

# These functions are described as follows:
# - LINEAR: maps input values in a specified interval [lower, upper] to the interval
# [0, 1]. Outside the specified interval, the function remains constant
# (that is, 0 or 1, depending on the side and selected mode (=decreasing or not))

# - TRIANGULAR: is 0 outside a specified interval and linearly increases to 1 from both
# interval ends, reaching the value 1 at the center of the interval
# This function is used per default for MATCH mode

# - BELL: A Gaussian bell curve, specified through the boundary values of the sigma
# interval, reaching the maximum value of 1 at the interval center

# For example we can define a third target working with the mode MATCH
# and a BELL bounds_transform_function.
# Note that the MATCH mode seeks to have the target at the mean between the two bounds.
# For example, choosing 95 and 105 will lead the algorithm seeking 100 as the optimal
# value. Thus, using the bounds, it is possible to control both the match target and
# the range around this target that is considered viable.

Target_3 = NumericalTarget(
    name="Target_3", mode="MATCH", bounds=(45, 55), bounds_transform_func="BELL"
)


# OBJECTIVE
# --------------------------------------------------------------------------------------


# Now to work with these three targets the objective object must be properly created
# The mode is set to 'DESIRABILITY' and the targets are described in a list

targets = [Target_1, Target_2, Target_3]

# for the recommender to work properly
# a combine_function is used to create a single target out of the several targets given.
# The combine function can either be the mean 'MEAN' or the geometric mean 'GEOM_MEAN'
# per default GEOM_MEAN is used
# Weights for each target can also be specified as a list of floats in the arguments
# Per default the weights are equally distributed between all targets.
# Also, weights are normalized internally, so it is not necessary to handle
# normalization or scaling here.


objective = Objective(
    mode="DESIRABILITY",
    targets=targets,
    weights=[20, 20, 60],
    combine_func="MEAN",
)

print(objective)


# BAYBE OBJECT
# --------------------------------------------------------------------------------------


baybe_obj = BayBE(searchspace=searchspace, objective=objective)

# This BayBE object can then be used to get recommendations and add measurements


# ITERATIONS EXAMPLE
# --------------------------------------------------------------------------------------


# The following loop performs some recommendations and add fake results
# and print what happens to internal data

N_ITERATIONS = 3

for kIter in range(N_ITERATIONS):
    print(f"\n\n##### ITERATION {kIter+1} #####")

    rec = baybe_obj.recommend(batch_quantity=3)
    print("\n### Recommended measurements:\n", rec)

    add_fake_results(rec, baybe_obj)
    print("\n### Recommended measurements with fake measured results:\n", rec)

    baybe_obj.add_measurements(rec)

    print("\n\n### Internal measurement dataframe computational representation Y:\n")
    print(baybe_obj.measurements_targets_comp)
