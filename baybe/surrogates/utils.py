"""Utilities for working with surrogates."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, TypeVar

from baybe.exceptions import InvalidSurrogateModelError
from baybe.surrogates.base import Surrogate

if TYPE_CHECKING:
    from botorch.posteriors import Posterior
    from torch import Tensor

    from baybe.surrogates.naive import MeanPredictionSurrogate

_TSurrogate = TypeVar("_TSurrogate", bound=Surrogate)


_constant_target_model_store: dict[int, MeanPredictionSurrogate] = {}
"""Dictionary for storing constant target fallback models. Keys are the IDs of the
surrogate models that temporarily have a fallback attached because they were
trained on constant training targets. Values are the corresponding fallback models."""


def catch_constant_targets(cls: type[Surrogate], std_threshold: float = 1e-6):
    """Make a ``Surrogate`` class robustly handle constant training targets.

    More specifically, "constant training targets" can mean either of:
        * The standard deviation of the training targets is below the given threshold.
        * There is only one target and the standard deviation cannot even be computed.

    The modified class handles the above cases separately from "regular operation"
    by resorting to a :class:`baybe.surrogates.naive.MeanPredictionSurrogate`,
    which is stored outside the model in a dictionary maintained by this decorator.

    Args:
        cls: The :class:`baybe.surrogates.base.Surrogate` to be augmented.
        std_threshold: The standard deviation threshold below which operation is
            switched to the alternative model.

    Returns:
        The modified class.
    """
    from baybe.surrogates.naive import MeanPredictionSurrogate

    # References to original methods
    _fit_original = cls._fit
    _posterior_original = cls._posterior

    def _posterior_new(self, candidates_comp_scaled: Tensor, /) -> Posterior:
        """Use fallback model if it exists, otherwise call original posterior."""
        # Alternative model fallback
        if constant_target_model := _constant_target_model_store.get(id(self), None):
            return constant_target_model._posterior(candidates_comp_scaled)

        # Regular operation
        return _posterior_original(self, candidates_comp_scaled)

    def _fit_new(self, train_x: Tensor, train_y: Tensor) -> None:
        """Original fit but with fallback model creation for constant targets."""
        if not (train_y.ndim == 2 and train_y.shape[-1] == 1):
            raise NotImplementedError(
                "The current logic is only implemented for single-target surrogates."
            )

        # Alternative model fallback
        if train_y.numel() == 1 or train_y.std() < std_threshold:
            model = MeanPredictionSurrogate()
            model._fit(train_x, train_y)
            _constant_target_model_store[id(self)] = model

        # Regular operation
        else:
            _constant_target_model_store.pop(id(self), None)
            _fit_original(self, train_x, train_y)

    # Replace the methods
    cls._posterior = _posterior_new  # type: ignore
    cls._fit = _fit_new  # type: ignore

    return cls


def batchify_mean_var_prediction(
    posterior: Callable[[_TSurrogate, Tensor], tuple[Tensor, Tensor]],
) -> Callable[[_TSurrogate, Tensor], tuple[Tensor, Tensor]]:
    """Wrap a posterior method to make it evaluate t-batches as an augmented q-batch."""

    @wraps(posterior)
    def sequential_posterior(
        model: _TSurrogate, candidates: Tensor
    ) -> tuple[Tensor, Tensor]:
        # If no batch dimensions are given, call the model directly
        if candidates.ndim == 2:
            return posterior(model, candidates)

        # Parameter batching is not (yet) supported
        if candidates.ndim > 3:
            raise ValueError("Multiple t-batch dimensions are not supported.")

        # Keep track of batch dimensions
        t_shape = candidates.shape[-3]
        q_shape = candidates.shape[-2]

        # Flatten the t-batch dimension into the q-batch dimension
        flattened = candidates.flatten(end_dim=-2)

        # Call the model on the entire input
        mean, var = posterior(model, flattened)

        # Restore the batch dimensions
        mean = mean.reshape((t_shape, q_shape))
        var = var.reshape((t_shape, q_shape))

        return mean, var

    return sequential_posterior


def batchify_ensemble_predictor(
    base_predictor: Callable[[Tensor], Tensor],
) -> Callable[[Tensor], Tensor]:
    """Wrap an ensemble predictor to make it evaluate t-batches as an augmented q-batch.

    Args:
        base_predictor: The ensemble predictor to be wrapped.

    Returns:
        The wrapped predictor.
    """

    @wraps(base_predictor)
    def batch_predictor(candidates: Tensor) -> Tensor:
        # If no batch dimensions are given, call the model directly
        if candidates.ndim == 2:
            return base_predictor(candidates)

        # Ensemble models do not (yet) support model parameter batching
        if candidates.ndim > 3:
            raise ValueError("Multiple t-batch dimensions are not supported.")

        # Keep track of batch dimensions
        t_shape = candidates.shape[-3]
        q_shape = candidates.shape[-2]

        # Flatten the t-batch dimension into the q-batch dimension
        flattened = candidates.flatten(end_dim=-2)

        # Call the model on the entire input
        predictions = base_predictor(flattened)

        # Assert that the model provides the ensemble predictions in the correct shape
        # (otherwise the reshaping operation below could silently produce wrong results)
        try:
            assert predictions.ndim == 2
            n_estimators = predictions.shape[0]
            assert predictions.shape[1] == t_shape * q_shape
        except AssertionError:
            raise InvalidSurrogateModelError(
                f"For the given input of shape {tuple(candidates.shape)}, "
                f"the ensemble model is supposed to create predictions of shape "
                f"(n_estimators, t_shape * q_shape) = "
                f"(n_estimators, {t_shape * q_shape}) "
                f"but returned an array of incompatible shape "
                f"{tuple(predictions.shape)}."
            )

        # Restore the batch dimensions
        predictions = predictions.reshape((n_estimators, t_shape, q_shape))
        predictions = predictions.permute((1, 0, 2))

        return predictions

    return batch_predictor
