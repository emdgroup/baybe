## Example for surrogate model with a custom architecture using `sklearn`

# This example shows how to define a `sklearn` model architecture and use it as a surrogate.
# Please note that the model is not designed to be useful but to demonstrate the workflow.

# This example assumes some basic familiarity with using BayBE.
# We thus refer to [`campaign`](./../Basics/campaign.md) for a basic example.

### Necessary imports

from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from torch import Tensor

from baybe.campaign import Campaign
from baybe.objective import Objective
from baybe.parameters import (
    CategoricalParameter,
    NumericalDiscreteParameter,
    SubstanceParameter,
)
from baybe.recommenders import (
    FPSRecommender,
    SequentialGreedyRecommender,
    TwoPhaseMetaRecommender,
)
from baybe.searchspace import SearchSpace
from baybe.surrogates import register_custom_architecture
from baybe.targets import NumericalTarget
from baybe.utils.dataframe import add_fake_results

### Surrogate Definition with BayBE Registration

# The final estimator class must follow the sklearn estimator interface.
# More details [here](https://scikit-learn.org/stable/developers/develop.html).

# The choice of using tensors in fit/predict is purely for BayBE, not a requirement.

# Final estimator


class MeanVarEstimator(BaseEstimator, RegressorMixin):
    """Stack final estimator for mean and variance."""

    def fit(self, data: Tensor, targets: Tensor) -> None:
        """No fit needed."""
        return

    def predict(self, data: Tensor) -> Tuple[Tensor, Tensor]:
        """Predict based on ensemble unweighted mean and variance."""
        mean = torch.tensor(data.mean(axis=1))
        var = torch.tensor(data.var(axis=1))
        return mean, var


# Registration

# The class must include `_fit` and `_posterior` functions with the correct signatures.


@register_custom_architecture(
    joint_posterior_attr=False, constant_target_catching=False, batchify_posterior=True
)
class StackingRegressorSurrogate:
    """Surrogate that extracts posterior from a stack of different regressors."""

    def __init__(self):
        self.model: Optional[StackingRegressor] = None

    def _posterior(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
        """See :class:`baybe.surrogates.Surrogate`."""
        return self.model.predict(candidates)

    def _fit(self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor) -> None:
        """See :class:`baybe.surrogates.Surrogate`."""
        estimators = [
            ("rf", RandomForestRegressor()),
            ("gb", GradientBoostingRegressor()),
            ("lr", LinearRegression()),
            ("rr", Ridge()),
        ]

        self.model = StackingRegressor(
            estimators=estimators,
            final_estimator=MeanVarEstimator(),
            cv=2,
        )

        self.model.fit(train_x, train_y.ravel())


### Experiment Setup

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


### Run DOE iterations with custom surrogate
# Create campaign

campaign = Campaign(
    searchspace=SearchSpace.from_product(parameters=parameters, constraints=None),
    objective=Objective(
        mode="SINGLE", targets=[NumericalTarget(name="Yield", mode="MAX")]
    ),
    recommender=TwoPhaseMetaRecommender(
        recommender=SequentialGreedyRecommender(
            surrogate_model=StackingRegressorSurrogate()
        ),
        initial_recommender=FPSRecommender(),
    ),
)

# Let's do a first round of recommendation
recommendation = campaign.recommend(batch_size=2)

print("Recommendation from campaign:")
print(recommendation)

# Add some fake results

add_fake_results(recommendation, campaign)
campaign.add_measurements(recommendation)

# Do another round of recommendations
recommendation = campaign.recommend(batch_size=2)

# Print second round of recommendations

print("Recommendation from campaign:")
print(recommendation)

print()


### Serialization

# Serialization of custom models is not supported

try:
    campaign.to_json()
except RuntimeError as e:
    print(f"Serialization Error Message: {e}")
