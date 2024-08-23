"""Linear surrogates.

Currently, the documentation for this surrogate is not available. This is due to a bug
in our documentation tool, see https://github.com/sphinx-doc/sphinx/issues/11750.

Since we plan to refactor the surrogates, this part of the documentation will be
available in the future. Thus, please have a look in the source code directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from attr import define, field
from sklearn.linear_model import ARDRegression

from baybe.surrogates.base import IndependentGaussianSurrogate
from baybe.surrogates.utils import batchify_mean_var_prediction, catch_constant_targets
from baybe.surrogates.validation import get_model_params_validator

if TYPE_CHECKING:
    from torch import Tensor


@catch_constant_targets
@define
class BayesianLinearSurrogate(IndependentGaussianSurrogate):
    """A Bayesian linear regression surrogate model."""

    supports_transfer_learning: ClassVar[bool] = False
    # See base class.

    model_params: dict[str, Any] = field(
        factory=dict,
        converter=dict,
        validator=get_model_params_validator(ARDRegression.__init__),
    )
    """Optional model parameter that will be passed to the surrogate constructor."""

    _model: ARDRegression | None = field(init=False, default=None, eq=False)
    """The actual model."""

    @batchify_mean_var_prediction
    def _estimate_moments(
        self, candidates_comp_scaled: Tensor, /
    ) -> tuple[Tensor, Tensor]:
        # See base class.

        # FIXME[typing]: It seems there is currently no better way to inform the type
        #   checker that the attribute is available at the time of the function call
        assert self._model is not None

        import torch

        # Get predictions
        dists = self._model.predict(candidates_comp_scaled.numpy(), return_std=True)

        # Split into posterior mean and variance
        mean = torch.from_numpy(dists[0])
        var = torch.from_numpy(dists[1]).pow(2)

        return mean, var

    def _fit(self, train_x: Tensor, train_y: Tensor) -> None:
        # See base class.
        self._model = ARDRegression(**(self.model_params))
        self._model.fit(train_x, train_y.ravel())
