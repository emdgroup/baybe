"""Naive surrogates."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Optional

from attr import define, field

from baybe.searchspace import SearchSpace
from baybe.surrogates.base import Surrogate
from baybe.surrogates.utils import batchify

if TYPE_CHECKING:
    from torch import Tensor


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
    _model: Optional[float] = field(init=False, default=None, eq=False)
    """The estimated posterior mean value of the training targets."""

    @batchify
    def _posterior(self, candidates: Tensor) -> tuple[Tensor, Tensor]:
        # See base class.

        import torch

        # TODO: use target value bounds for covariance scaling when explicitly provided
        mean = self._model * torch.ones([len(candidates)])
        var = torch.ones(len(candidates))
        return mean, var

    def _fit(self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor) -> None:
        # See base class.
        self._model = train_y.mean().item()
