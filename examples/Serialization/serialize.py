"""Simple serialization example."""

from baybe.core import BayBE
from baybe.parameters import Categorical, GenericSubstance, NumericDiscrete
from baybe.searchspace import SearchSpace
from baybe.strategies.bayesian import SequentialGreedyRecommender
from baybe.strategies.sampling import FPSRecommender
from baybe.strategies.strategy import Strategy
from baybe.targets import NumericalTarget, Objective

# Define the experimental parameters
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
    GenericSubstance(
        name="Solvent",
        data={"Solvent A": "COC", "Solvent B": "CCCCC"},
        encoding="MORDRED",
    ),
]

# Define the measured target variables
targets = [NumericalTarget(name="Yield", mode="MAX")]

# Create the DOE object
baybe = BayBE(
    searchspace=SearchSpace.create(parameters=parameters, constraints=None),
    objective=Objective(mode="SINGLE", targets=targets),
    strategy=Strategy(
        recommender=SequentialGreedyRecommender(),
        initial_recommender=FPSRecommender(),
    ),
)
print(f"{'#'*30} Original object {'#'*30}")
print(baybe, end="\n" * 3)

# Serialize to object to JSON
string = baybe.to_json()
print(f"{'#'*30} JSON string {'#'*30}")
print(string, end="\n" * 3)

# Deserialize the JSON string back to an object
print(f"{'#'*30} Deserialized object {'#'*30}")
baybe2 = BayBE.from_json(string)
print(baybe2, end="\n" * 3)

# Assert that both objects are equal
assert baybe == baybe2
