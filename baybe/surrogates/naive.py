"""Naive surrogates."""

from typing import ClassVar, Optional, Tuple

import torch
from attr import define, field
from torch import Tensor

from baybe.searchspace import SearchSpace
from baybe.surrogates.base import Surrogate
from baybe.surrogates.utils import batchify


@define
class MeanPredictionSurrogate(Surrogate):
    """A trivial surrogate model.

    It provides the average value of the training targets
    as posterior mean and a (data-independent) constant posterior variance.
    """

    # Class variables
    joint_posterior: ClassVar[bool] = False
    # See base class.

    supports_transfer_learning: ClassVar[bool] = False
    # See base class.

    # Object variables
    target_value: Optional[float] = field(init=False, default=None)
    """The value of the posterior mean."""

    @batchify
    def _posterior(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
        # See base class.
        # TODO: use target value bounds for covariance scaling when explicitly provided
        mean = self.target_value * torch.ones([len(candidates)])
        var = torch.ones(len(candidates))
        return mean, var

    def _fit(self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor) -> None:
        # See base class.
        self.target_value = train_y.mean().item()
