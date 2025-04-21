"""NGBoost surrogates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from attrs import define, field
from typing_extensions import override

from baybe.parameters.base import Parameter
from baybe.surrogates.base import IndependentGaussianSurrogate
from baybe.surrogates.utils import batchify_mean_var_prediction, catch_constant_targets
from baybe.surrogates.validation import get_model_params_validator
from baybe.utils.conversion import to_string

if TYPE_CHECKING:
    from botorch.models.transforms.input import InputTransform
    from botorch.models.transforms.outcome import OutcomeTransform
    from torch import Tensor


@catch_constant_targets
@define
class NGBoostSurrogate(IndependentGaussianSurrogate):
    """A natural-gradient-boosting surrogate model."""

    supports_transfer_learning: ClassVar[bool] = False
    # See base class.

    _default_model_params: ClassVar[dict] = {"n_estimators": 25, "verbose": False}
    """Class variable encoding the default model parameters."""

    model_params: dict[str, Any] = field(factory=dict, converter=dict)
    """Optional model parameter that will be passed to the surrogate constructor."""

    # TODO: Type should be `NGBRegressor | None` but is currently
    #   omitted due to: https://github.com/python-attrs/cattrs/issues/531
    _model = field(init=False, default=None, eq=False)
    """The actual model."""

    @model_params.validator
    def _validate_model_params(self, attr, value) -> None:
        from baybe._optional.ngboost import NGBRegressor

        validator = get_model_params_validator(NGBRegressor.__init__)
        validator(self, attr, value)

    def __attrs_post_init__(self):
        self.model_params = {**self._default_model_params, **self.model_params}

    @override
    @staticmethod
    def _make_parameter_scaler_factory(
        parameter: Parameter,
    ) -> type[InputTransform] | None:
        # Tree-like models do not require any input scaling
        return None

    @override
    @staticmethod
    def _make_target_scaler_factory() -> type[OutcomeTransform] | None:
        # Tree-like models do not require any output scaling
        return None

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
        dists = self._model.pred_dist(candidates_comp_scaled)

        # Split into posterior mean and variance
        mean = torch.from_numpy(dists.mean())
        var = torch.from_numpy(dists.var)

        return mean, var

    @override
    def _fit(self, train_x: Tensor, train_y: Tensor) -> None:
        from baybe._optional.ngboost import NGBRegressor

        self._model = NGBRegressor(**(self.model_params)).fit(train_x, train_y.ravel())

    @override
    def __str__(self) -> str:
        fields = [to_string("Model Params", self.model_params, single_line=True)]
        return to_string(super().__str__(), *fields)
