"""Random forest surrogates."""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

import numpy as np
from attr import define, field
from sklearn.ensemble import RandomForestRegressor

from baybe.parameters.base import Parameter
from baybe.surrogates.base import Surrogate
from baybe.surrogates.utils import batchify_ensemble_predictor, catch_constant_targets
from baybe.surrogates.validation import get_model_params_validator
from baybe.utils.plotting import to_string

if TYPE_CHECKING:
    from botorch.models.ensemble import EnsemblePosterior
    from botorch.models.transforms.input import InputTransform
    from botorch.models.transforms.outcome import OutcomeTransform
    from torch import Tensor


class _Predictor(Protocol):
    """A basic predictor."""

    def predict(self, x: np.ndarray, /) -> np.ndarray: ...


@catch_constant_targets
@define
class RandomForestSurrogate(Surrogate):
    """A random forest surrogate model."""

    supports_transfer_learning: ClassVar[bool] = False
    # See base class.

    model_params: dict[str, Any] = field(
        factory=dict,
        converter=dict,
        validator=get_model_params_validator(RandomForestRegressor.__init__),
    )
    """Optional model parameter that will be passed to the surrogate constructor."""

    _model: RandomForestRegressor | None = field(init=False, default=None, eq=False)
    """The actual model."""

    @staticmethod
    def _make_parameter_scaler_factory(
        parameter: Parameter,
    ) -> type[InputTransform] | None:
        # See base class.

        # Tree-like models do not require any input scaling
        return None

    @staticmethod
    def _make_target_scaler_factory() -> type[OutcomeTransform] | None:
        # See base class.

        # Tree-like models do not require any output scaling
        return None

    def _posterior(self, candidates_comp_scaled: Tensor, /) -> EnsemblePosterior:
        # See base class.

        from botorch.models.ensemble import EnsemblePosterior

        @batchify_ensemble_predictor
        def predict(candidates_comp_scaled: Tensor) -> Tensor:
            """Make the end-to-end ensemble prediction."""
            import torch

            # FIXME[typing]: It seems there is currently no better way to inform the
            #   type checker that the attribute is available at the time of the
            #   function call
            assert self._model is not None

            return torch.from_numpy(
                self._predict_ensemble(
                    self._model.estimators_, candidates_comp_scaled.numpy()
                )
            )

        return EnsemblePosterior(predict(candidates_comp_scaled).unsqueeze(-1))

    @staticmethod
    def _predict_ensemble(
        predictors: Collection[_Predictor], candidates: np.ndarray
    ) -> np.ndarray:
        """Evaluate an ensemble of predictors on a given candidate set."""
        # Extract shapes
        n_candidates = len(candidates)
        n_estimators = len(predictors)

        # Evaluate all trees
        predictions = np.zeros((n_estimators, n_candidates))
        for p, predictor in enumerate(predictors):
            predictions[p] = predictor.predict(candidates)

        return predictions

    def _fit(self, train_x: Tensor, train_y: Tensor) -> None:
        # See base class.
        self._model = RandomForestRegressor(**(self.model_params))
        self._model.fit(train_x.numpy(), train_y.numpy().ravel())

    def __str__(self) -> str:
        fields = [to_string("Model Params", self.model_params, single_line=True)]
        return to_string(super().__str__(), *fields)
