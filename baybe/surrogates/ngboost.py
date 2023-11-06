"""NGBoost surrogates."""

from typing import Any, ClassVar, Dict, Optional, Tuple

import torch
from attr import define, field
from ngboost import NGBRegressor
from torch import Tensor

from baybe.searchspace import SearchSpace
from baybe.surrogates.base import Surrogate
from baybe.surrogates.utils import batchify, catch_constant_targets, scale_model
from baybe.surrogates.validation import get_model_params_validator


@catch_constant_targets
@scale_model
@define
class NGBoostSurrogate(Surrogate):
    """A natural-gradient-boosting surrogate model.

    Args:
        model_params: Optional model parameters.
    """

    # Class variables
    joint_posterior: ClassVar[bool] = False
    supports_transfer_learning: ClassVar[bool] = False
    _default_model_params: ClassVar[dict] = {"n_estimators": 25, "verbose": False}
    """Class variable encoding the default model parameters."""

    # Object variables
    model_params: Dict[str, Any] = field(
        factory=dict,
        converter=dict,
        validator=get_model_params_validator(NGBRegressor.__init__),
    )
    _model: Optional[NGBRegressor] = field(init=False, default=None)

    def __attrs_post_init__(self):
        self.model_params = {**self._default_model_params, **self.model_params}

    @batchify
    def _posterior(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
        # See base class. pylint:disable=missing-function-docstring
        # Get predictions
        dists = self._model.pred_dist(candidates)

        # Split into posterior mean and variance
        mean = torch.from_numpy(dists.mean())
        var = torch.from_numpy(dists.var)

        return mean, var

    def _fit(self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor) -> None:
        # See base class. pylint:disable=missing-function-docstring
        self._model = NGBRegressor(**(self.model_params)).fit(train_x, train_y.ravel())
