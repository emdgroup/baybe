"""Fitting criteria for the Gaussian process surrogate."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import pandas as pd
from attrs import define
from typing_extensions import override

from baybe.objectives.base import Objective

if TYPE_CHECKING:
    from gpytorch.likelihoods import Likelihood as GPyTorchLikelihood
    from gpytorch.mlls import MarginalLogLikelihood
    from gpytorch.models import GP as GPyTorchModel

    from baybe.searchspace.core import SearchSpace


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


@define
class _MLLForNonTLFitCriterionFactory(FitCriterionFactoryProtocol):
    """A fit criterion factory switching between MLL and BayBE default.

    In transfer learning contexts, delegates to
    :class:`baybe.surrogates.gaussian_process.presets.baybe.BayBEFitCriterionFactory`.
    """

    @override
    def __call__(
        self, searchspace: SearchSpace, objective: Objective, measurements: pd.DataFrame
    ) -> FitCriterion:
        if searchspace.task_idx is None:
            return FitCriterion.MARGINAL_LOG_LIKELIHOOD

        from baybe.surrogates.gaussian_process.presets.baybe import (
            BayBEFitCriterionFactory,
        )

        return BayBEFitCriterionFactory()(searchspace, objective, measurements)
