### Custom Model example based on a Stacked Regressor
# pylint: disable=unused-argument

"""
This example shows the creation of a Stacked Regressor
using `sklearn`, and the registration of the architecture for BayBE.
"""

# Please note that this is an example architecture
# The model is not designed to be useful but to demonstrate the workflow

#### Necessary imports

from typing import Optional, Tuple

from baybe.searchspace import SearchSpace

from baybe.surrogate import register_custom_architecture

from sklearn.base import BaseEstimator, RegressorMixin

# Stacked models + estimators to be stacked
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)

from sklearn.linear_model import LinearRegression, Ridge
from torch import Tensor


#### Define final estimator


class MeanVarEstimator(BaseEstimator, RegressorMixin):
    """
    Final estimator in the stack that returns
    the unweighted mean and variance across
    the ensemble of estimators
    """

    def fit(self, data: Tensor, targets: Tensor) -> None:
        """No fit needed"""
        return

    def predict(self, data: Tensor) -> Tuple[Tensor]:
        """Predict based on ensemble mean and var"""
        mean = Tensor(data.mean(axis=1))
        var = Tensor(data.var(axis=1))
        return mean, var


#### Surrogate Definition with BayBE Registration

# The class must include `_fit` and `_posterior` functions with the correct signatures

# Registration
@register_custom_architecture(
    joint_posterior_attr=False, constant_target_catching=False, batchify_posterior=True
)
class StackingRegressorSurrogate:
    """
    Function definitions for a surrogate model that
    extracts posterior from a stack of different regressors.
    """

    def __init__(self):
        self.model: Optional[StackingRegressor] = None

    def _posterior(self, candidates: Tensor) -> Tuple[Tensor]:
        """See baybe.surrogate.Surrogate."""
        return self.model.predict(candidates)

    def _fit(self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor) -> None:
        """See baybe.surrogate.Surrogate."""
        estimators = [
            ("rf", RandomForestRegressor()),
            ("gb", GradientBoostingRegressor()),
            ("lr", LinearRegression()),
            ("rr", Ridge()),
        ]

        self.model = StackingRegressor(
            estimators=estimators, final_estimator=MeanVarEstimator(), cv=2  # 2 fold cv
        )

        self.model.fit(train_x, train_y.ravel())
