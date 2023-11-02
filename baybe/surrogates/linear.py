"""Linear surrogates."""

from typing import Any, ClassVar, Dict, Optional, Tuple

import torch
from attr import define, field
from sklearn.linear_model import ARDRegression
from torch import Tensor

from baybe.searchspace import SearchSpace
from baybe.surrogates.base import Surrogate
from baybe.surrogates.utils import (
    _get_model_params_validator,
    batchify,
    catch_constant_targets,
    scale_model,
)


@catch_constant_targets
@scale_model
@define
class BayesianLinearSurrogate(Surrogate):
    """A Bayesian linear regression surrogate model.

    Args:
        model_params: Optional model parameters.
    """

    # Class variables
    joint_posterior: ClassVar[bool] = False
    supports_transfer_learning: ClassVar[bool] = False

    # Object variables
    model_params: Dict[str, Any] = field(
        factory=dict,
        converter=dict,
        validator=_get_model_params_validator(ARDRegression.__init__),
    )
    _model: Optional[ARDRegression] = field(init=False, default=None)

    @batchify
    def _posterior(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
        # See base class.
        # Get predictions
        dists = self._model.predict(candidates.numpy(), return_std=True)

        # Split into posterior mean and variance
        mean = torch.from_numpy(dists[0])
        var = torch.from_numpy(dists[1]).pow(2)

        return mean, var

    def _fit(self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor) -> None:
        # See base class.
        self._model = ARDRegression(**(self.model_params))
        self._model.fit(train_x, train_y.ravel())
