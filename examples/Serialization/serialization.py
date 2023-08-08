"""Simple serialization example.

This example shows the creation of a BayBE object, how to serialize it and also
de-serialize it and shows that the "original" and "new" objects behave the same.
It also shows an additional way of creating BayBE objects via config files and
how to validate these.

Note that this example does not explain the basics of object creation, we refer to the
basics example for explanations on this.
"""

import numpy as np

from baybe import BayBE
from baybe.parameters import (
    CategoricalParameter,
    NumericalDiscreteParameter,
    SubstanceParameter,
)
from baybe.searchspace import SearchSpace
from baybe.strategies import FPSRecommender, SequentialGreedyRecommender, Strategy
from baybe.targets import NumericalTarget, Objective

# We start by defining the parameters of the experiment.
parameters = [
    CategoricalParameter(
        name="Granularity",
        values=["coarse", "medium", "fine"],
        encoding="OHE",
    ),
    NumericalDiscreteParameter(
        name="Pressure[bar]",
        values=[1, 5, 10],
        tolerance=0.2,
    ),
    NumericalDiscreteParameter(
        name="Temperature[degree_C]",
        values=np.linspace(100, 200, 10),
    ),
    SubstanceParameter(
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
    searchspace=SearchSpace.from_product(parameters=parameters, constraints=None),
    objective=Objective(mode="SINGLE", targets=targets),
    strategy=Strategy(
        recommender=SequentialGreedyRecommender(),
        initial_recommender=FPSRecommender(),
    ),
)
# We begin by printing the original BayBE object
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

# To further show how serialization affects working with BayBE objects, we will now
# create and compare some recommendations in both bayBE objects.

# We now do recommendations for both objects and print them
recommendation_orig = baybe_orig.recommend(batch_quantity=2)
recommendation_recreate = baybe_recreate.recommend(batch_quantity=2)

print("Recommendation from original object:")
print(recommendation_orig)

print("Recommendation from recreated object:")
print(recommendation_recreate)
