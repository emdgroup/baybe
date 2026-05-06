"""Optimization criteria for the Gaussian process surrogate."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from baybe.surrogates.gaussian_process.components.generic import (
    GPComponentFactoryProtocol,
    PlainGPComponentFactory,
)

if TYPE_CHECKING:
    from gpytorch.likelihoods import Likelihood as GPyTorchLikelihood
    from gpytorch.mlls import MarginalLogLikelihood
    from gpytorch.models import GP as GPyTorchModel


class Criterion(Enum):
    """Available optimization criteria for GP hyperparameter selection."""

    MARGINAL_LOG_LIKELIHOOD = "MARGINAL_LOG_LIKELIHOOD"
    """Exact marginal log-likelihood."""

    LEAVE_ONE_OUT = "LEAVE_ONE_OUT"
    """Leave-one-out cross-validation pseudo-likelihood."""

    def to_gpytorch(
        self, likelihood: GPyTorchLikelihood, model: GPyTorchModel
    ) -> MarginalLogLikelihood:
        """Create the corresponding GPyTorch MLL object."""
        import gpytorch

        mll_class = {
            Criterion.MARGINAL_LOG_LIKELIHOOD: gpytorch.ExactMarginalLogLikelihood,
            Criterion.LEAVE_ONE_OUT: gpytorch.mlls.LeaveOneOutPseudoLikelihood,
        }[self]
        return mll_class(likelihood, model)


CriterionFactoryProtocol = GPComponentFactoryProtocol[Criterion]
"""A protocol defining the interface for criterion factories."""

PlainCriterionFactory = PlainGPComponentFactory[Criterion]
"""A trivial factory that returns a fixed criterion."""
