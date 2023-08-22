### Example for surrogate model with a custom architecture

"""
This example shows the creation of a BayBE object,
and the usage of pre-registered surrogate models with custom architecture in BayBE.
"""

# This example assumes some basic familiarity with using BayBE.
# We thus refer to [`baybe_object`](./../Basics/baybe_object.md) for a basic example.

#### Necessary imports

import numpy as np

from baybe.core import BayBE
from baybe.parameters import (
    CategoricalParameter,
    NumericalDiscreteParameter,
    SubstanceParameter,
)
from baybe.searchspace import SearchSpace
from baybe.strategies import FPSRecommender, SequentialGreedyRecommender, Strategy

from baybe.targets import NumericalTarget, Objective
from baybe.utils import add_fake_results

from dropout_surrogate import NeuralNetDropoutSurrogate
from stacking_surrogate import StackingRegressorSurrogate


#### Experiment Setup

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

#### Initialize a custom surrogate architecture class

# Please note this is based on the [`MCDO`](./../Model_Settings/dropout_surrogate.md) architecture
# as well as the [`Stacked`](./../Model_Settings/stacking_surrogate.md) architecture
# The class can also take in additional parameters if needed
surrogate_models = [NeuralNetDropoutSurrogate(), StackingRegressorSurrogate()]

#### BayBE interaction to experiment with surrogates


def test_surrogate_in_baybe(surrogate_model):
    """Runs a simple baybe loop with a model."""
    print(f"Running BayBE with {surrogate_model.model.__class__.__name__}:")

    # Create BayBE Object
    baybe_obj = BayBE(
        searchspace=SearchSpace.from_product(parameters=parameters, constraints=None),
        objective=Objective(
            mode="SINGLE", targets=[NumericalTarget(name="Yield", mode="MAX")]
        ),
        strategy=Strategy(
            recommender=SequentialGreedyRecommender(surrogate_model=surrogate_model),
            initial_recommender=FPSRecommender(),
        ),
    )

    # Let's do a first round of recommendation
    recommendation = baybe_obj.recommend(batch_quantity=2)

    print("Recommendation from baybe object:")
    print(recommendation)

    # Add some fake results
    add_fake_results(recommendation, baybe_obj)
    baybe_obj.add_measurements(recommendation)

    # Do another round of recommendations
    recommendation = baybe_obj.recommend(batch_quantity=2)

    # Print second round of recommendations
    print("Recommendation from baybe object:")
    print(recommendation)

    print()


#### Run BayBE with custom surrogates
for surrogate in surrogate_models:
    test_surrogate_in_baybe(surrogate)


#### Serialization

# Create BayBE Object for serialization
baybe_test = BayBE(
    searchspace=SearchSpace.from_product(parameters=parameters, constraints=None),
    objective=Objective(
        mode="SINGLE", targets=[NumericalTarget(name="Yield", mode="MAX")]
    ),
    strategy=Strategy(
        recommender=SequentialGreedyRecommender(surrogate_model=surrogate_models[0]),
        initial_recommender=FPSRecommender(),
    ),
)
# Serialization of custom models is not supported
try:
    baybe_test.to_json()
except RuntimeError as e:
    print()
    print(f"Serialization Error Message: {e}")
    print()
