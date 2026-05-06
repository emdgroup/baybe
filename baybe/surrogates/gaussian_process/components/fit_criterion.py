"""Fitting criteria for the Gaussian process surrogate."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpytorch.likelihoods import Likelihood as GPyTorchLikelihood
    from gpytorch.mlls import MarginalLogLikelihood
    from gpytorch.models import GP as GPyTorchModel


class FitCriterion(Enum):
    """Available fitting criteria for GP hyperparameter optimization."""

    MARGINAL_LOG_LIKELIHOOD = "MARGINAL_LOG_LIKELIHOOD"
    """Exact marginal log-likelihood."""

    LEAVE_ONE_OUT_PSEUDOLIKELIHOOD = "LEAVE_ONE_OUT_PSEUDOLIKELIHOOD"
    """Leave-one-out cross-validation pseudo-likelihood."""

    def to_gpytorch(
        self, likelihood: GPyTorchLikelihood, model: GPyTorchModel
    ) -> MarginalLogLikelihood:
        """Create the corresponding GPyTorch MLL object."""
        import gpytorch

        mll_class = {
            FitCriterion.MARGINAL_LOG_LIKELIHOOD: gpytorch.ExactMarginalLogLikelihood,
            FitCriterion.LEAVE_ONE_OUT_PSEUDOLIKELIHOOD: gpytorch.mlls.LeaveOneOutPseudoLikelihood,  # noqa: E501
        }[self]
        return mll_class(likelihood, model)


# Delayed import to avoid circular dependency
from baybe.surrogates.gaussian_process.components.generic import (  # noqa: E402
    GPComponentFactoryProtocol,
    PlainGPComponentFactory,
)

FitCriterionFactoryProtocol = GPComponentFactoryProtocol[FitCriterion]
"""A protocol defining the interface for fit criterion factories."""

PlainFitCriterionFactory = PlainGPComponentFactory[FitCriterion]
"""A trivial factory that returns a fixed fit criterion."""
