"""Linear surrogates."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, ClassVar, TypedDict

from attrs import define, field
from typing_extensions import override

from baybe.surrogates.base import IndependentGaussianSurrogate
from baybe.surrogates.utils import batchify_mean_var_prediction, catch_constant_targets
from baybe.surrogates.validation import make_dict_validator
from baybe.utils.conversion import to_string

if TYPE_CHECKING:
    from torch import Tensor


class _ARDRegressionParams(TypedDict, total=False):
    """Optional ARDRegression parameters.

    See :class:`~sklearn.linear_model.ARDRegression`.
    """

    max_iter: int
    tol: float
    alpha_1: float
    alpha_2: float
    lambda_1: float
    lambda_2: float
    compute_score: bool
    threshold_lambda: float
    fit_intercept: bool
    copy_X: bool
    verbose: bool


@catch_constant_targets
@define
class BayesianLinearSurrogate(IndependentGaussianSurrogate):
    """A Bayesian linear regression surrogate model."""

    supports_transfer_learning: ClassVar[bool] = False
    # See base class.

    model_params: _ARDRegressionParams = field(
        factory=dict,
        converter=dict,
        validator=make_dict_validator(_ARDRegressionParams),
    )
    """Optional model parameter that will be passed to the surrogate constructor.

    For allowed keys and values, see :class:`~sklearn.linear_model.ARDRegression`.
    """

    # TODO: type should be `ARDRegression | None` but is currently omitted due to:
    #  https://github.com/python-attrs/cattrs/issues/531
    _model = field(init=False, default=None, eq=False)
    """The actual model."""

    @override
    @batchify_mean_var_prediction
    def _estimate_moments(
        self, candidates_comp_scaled: Tensor, /
    ) -> tuple[Tensor, Tensor]:
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

    @override
    def _fit(self, train_x: Tensor, train_y: Tensor) -> None:
        from sklearn.linear_model import ARDRegression

        self._model = ARDRegression(**(self.model_params))
        self._model.fit(train_x, train_y.ravel())

    @override
    def __str__(self) -> str:
        fields = [to_string("Model Params", self.model_params, single_line=True)]
        return to_string(super().__str__(), *fields)


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
