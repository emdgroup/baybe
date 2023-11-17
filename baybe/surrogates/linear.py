"""Linear surrogates.

Currently, the documentation for this surrogate is not available. This is due to a bug
in our documentation tool, see https://github.com/sphinx-doc/sphinx/issues/11750.

Since we plan to refactor the surrogates, this part of the documentation will be
available in the future. Thus, please have a look in the source code directly.
"""

from typing import Any, ClassVar, Dict, Optional, Tuple

import torch
from attr import define, field
from sklearn.linear_model import ARDRegression
from torch import Tensor

from baybe.searchspace import SearchSpace
from baybe.surrogates.base import Surrogate
from baybe.surrogates.utils import batchify, catch_constant_targets, scale_model
from baybe.surrogates.validation import get_model_params_validator


@catch_constant_targets
@scale_model
@define
class BayesianLinearSurrogate(Surrogate):
    """A Bayesian linear regression surrogate model."""

    # Class variables
    joint_posterior: ClassVar[bool] = False
    # See base class.

    supports_transfer_learning: ClassVar[bool] = False
    # See base class.

    # Object variables
    model_params: Dict[str, Any] = field(
        factory=dict,
        converter=dict,
        validator=get_model_params_validator(ARDRegression.__init__),
    )
    # See base class.

    _model: Optional[ARDRegression] = field(init=False, default=None)
    """The actual model."""

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
