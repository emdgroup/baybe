"""Random forest surrogates."""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol, TypedDict

import numpy as np
import numpy.typing as npt
from attrs import define, field
from numpy.random import RandomState
from typing_extensions import override

from baybe.parameters.base import Parameter
from baybe.surrogates.base import Surrogate
from baybe.surrogates.utils import batchify_ensemble_predictor, catch_constant_targets
from baybe.surrogates.validation import make_dict_validator
from baybe.utils.conversion import to_string

if TYPE_CHECKING:
    from botorch.models.ensemble import EnsemblePosterior
    from botorch.models.transforms.input import InputTransform
    from botorch.models.transforms.outcome import OutcomeTransform
    from torch import Tensor


class _RandomForestRegressorParams(TypedDict, total=False):
    """Optional RandomForestRegressor parameters.

    See :class:`~sklearn.ensemble.RandomForestRegressor`.
    """

    n_estimators: int
    criterion: Literal["squared_error", "absolute_error", "friedman_mse", "poisson"]
    max_depth: int
    min_samples_split: int | float
    min_samples_leaf: int | float
    min_weight_fraction_leaf: float
    max_features: Literal["sqrt", "log2"] | int | float | None
    max_leaf_nodes: int | None
    min_impurity_decrease: float
    bootstrap: bool
    oob_score: bool
    n_jobs: int | None
    random_state: int | RandomState | None
    verbose: int
    warm_start: bool
    ccp_alpha: float
    max_samples: int | float | None
    monotonic_cst: npt.ArrayLike | int | None


class _Predictor(Protocol):
    """A basic predictor."""

    def predict(self, x: np.ndarray, /) -> np.ndarray: ...


@catch_constant_targets
@define
class RandomForestSurrogate(Surrogate):
    """A random forest surrogate model."""

    supports_transfer_learning: ClassVar[bool] = False
    # See base class.

    model_params: _RandomForestRegressorParams = field(
        factory=dict,
        converter=dict,
        validator=make_dict_validator(_RandomForestRegressorParams),
    )
    """Optional model parameter that will be passed to the surrogate constructor.

    For allowed keys and values, see :class:`~sklearn.ensemble.RandomForestRegressor`.
    """

    # TODO: type should be `RandomForestRegressor | None` but is currently omitted due
    #  to: https://github.com/python-attrs/cattrs/issues/531
    _model = field(init=False, default=None, eq=False)
    """The actual model."""

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
    def _posterior(self, candidates_comp_scaled: Tensor, /) -> EnsemblePosterior:
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

    @override
    def _fit(self, train_x: Tensor, train_y: Tensor) -> None:
        from sklearn.ensemble import RandomForestRegressor

        self._model = RandomForestRegressor(**(self.model_params))
        self._model.fit(train_x.numpy(), train_y.numpy().ravel())

    @override
    def __str__(self) -> str:
        fields = [to_string("Model Params", self.model_params, single_line=True)]
        return to_string(super().__str__(), *fields)
