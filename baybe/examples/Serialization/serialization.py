"""Simple serialization example.

This exampls shows the creation of a baybe object, how to serialize it and also
de-serialize it and shows that the "original" and "new" objects behave the same.
It also shows an additional way of creating baybe objects via config files and
how to validate these.

Note that this example does not explain the basics of object creation, we refer to the
basics example for explanations on this.
"""

import numpy as np

from baybe.core import BayBE
from baybe.parameters import Categorical, GenericSubstance, NumericDiscrete
from baybe.searchspace import SearchSpace
from baybe.strategies.bayesian import SequentialGreedyRecommender
from baybe.strategies.sampling import FPSRecommender
from baybe.strategies.strategy import Strategy
from baybe.targets import NumericalTarget, Objective
from baybe.utils import add_fake_results

# We start by defining the parameters of the experiment.
parameters = [
    Categorical(
        name="Granularity",
        values=["coarse", "medium", "fine"],
        encoding="OHE",
    ),
    NumericDiscrete(
        name="Pressure[bar]",
        values=[1, 5, 10],
        tolerance=0.2,
    ),
    NumericDiscrete(
        name="Temperature[degree_C]",
        values=np.linspace(100, 200, 10).tolist(),
    ),
    GenericSubstance(
        name="Solvent",
        data={
            "Solvent A": "COC",
            "Solvent B": "CCC",
            "Solvent C": "O",
            "Solvent D": "CS(=O)C",
        },
        encoding="MORDRED",
    ),
]

# Define a single target
targets = [NumericalTarget(name="Yield", mode="MAX")]

# Create the DOE object
baybe_orig = BayBE(
    searchspace=SearchSpace.create(parameters=parameters, constraints=None),
    objective=Objective(mode="SINGLE", targets=targets),
    strategy=Strategy(
        recommender=SequentialGreedyRecommender(),
        initial_recommender=FPSRecommender(),
    ),
)
# We begin by printing the original baybe object
print(f"{'#'*30} Original object {'#'*30}")
print(baybe_orig, end="\n" * 3)

# We next serialize the BayBE object to JSON
# This yields a JSON representation that we also print here for convenience.
string = baybe_orig.to_json()
print(f"{'#'*30} JSON string {'#'*30}")
print(string, end="\n" * 3)

# Deserialize the JSON string back to an object
print(f"{'#'*30} Deserialized object {'#'*30}")
baybe_recreate = BayBE.from_json(string)
print(baybe_recreate, end="\n" * 3)

# Verify that both objects are equal
assert baybe_orig == baybe_recreate
print("Passed basic assertion check!")

# To further show how serialization affects working with baybe objects, we will now
# create and store some recommendations, then perform a serialization and a
# de-serialization and show that the next recommendation will be the same.

# We thus begin by running some iterations with fake results
for _ in range(5):
    recommendation = baybe_orig.recommend(batch_quantity=2)
    add_fake_results(recommendation, baybe_orig)
    baybe_orig.add_measurements(recommendation)

# We re-print the baybe object after these recommendation
print(f"{'#'*30} Original object after some recommendations {'#'*30}")
print(baybe_orig, end="\n" * 3)

# We now serialize and de-serialize it and check for equality again.
json_after_recommendations = baybe_orig.to_json()
baybe_recreate_after_rec = BayBE.from_json(json_after_recommendations)
assert baybe_orig == baybe_recreate_after_rec
print("Passed assertion check after some recommendations!")

# We now do recommendations for both objects and show that they are equal
recommendation_orig = baybe_orig.recommend(batch_quantity=2)
recommendation_recreate = baybe_recreate_after_rec.recommend(batch_quantity=2)

assert recommendation_orig.equals(recommendation_recreate)
print("Passed assertion check for recommendations!")
