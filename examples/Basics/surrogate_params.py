"""Surrogate Model Parameter example.

This example shows the creation of a BayBE object, how to define surrogate
models with custom model parameters and the validations that are done.
It also shows how to specify these parameters through a configuration.

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
from baybe.surrogate import NGBoostModel
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
        values=np.linspace(100, 200, 10),
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


# Surrogate Model with custom parameters
surrogate_model = NGBoostModel(model_params={"n_estimators": 50, "verbose": True})

# See the following links for available options in each model

# NOTE. GaussianProcessModel will support custom parameters in the future

# RandomForestModel
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

# NGBoostModel
# https://stanfordmlgroup.github.io/ngboost/1-useage.html

# BayesianLinearModel
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html


# Create the DOE object
baybe_obj = BayBE(
    searchspace=SearchSpace.from_product(parameters=parameters, constraints=None),
    objective=Objective(mode="SINGLE", targets=targets),
    strategy=Strategy(
        recommender=SequentialGreedyRecommender(surrogate_model=surrogate_model),
        initial_recommender=FPSRecommender(),
    ),
)


# We can print the surrogate model object
print(f"{'#'*30} BayBE object {'#'*30}")
print(surrogate_model.to_json(), end="\n" * 3)


# Let's do a first round of recommendation
recommendation = baybe_obj.recommend(batch_quantity=2)

print("Recommendation from baybe object:")
print(recommendation)


# Add some fake results
add_fake_results(recommendation, baybe_obj)
baybe_obj.add_measurements(recommendation)


# Model outputs
# NOTE. This model is only triggered when there is data.
print()
print("Here you will see some model outputs as we set verbose to True")
print(f"{'#'*60}")

# Do another round of recommendations
recommendation = baybe_obj.recommend(batch_quantity=2)

print(f"{'#'*60}")
print()


# Second round of recommendations
print("Recommendation from baybe object:")
print(recommendation)


# By configuration:
# NOTE. This can be placed inside an overall baybe config
# See examples/Serialization/create_from_config for an example
CONFIG = """
{
    "type": "NGBoostModel",
    "model_params": {
        "n_estimators": 50,
        "verbose": true
    }

}
"""

# Create a model based on the json string
recreate_model = NGBoostModel.from_json(CONFIG)

# Ensure they are equal
assert recreate_model == surrogate_model
