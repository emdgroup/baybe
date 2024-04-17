"""Utilities for working with surrogates."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Callable

from baybe.scaler import DefaultScaler
from baybe.searchspace import SearchSpace

if TYPE_CHECKING:
    from torch import Tensor

    from baybe.surrogates.base import Surrogate


def _prepare_inputs(x: Tensor) -> Tensor:
    """Validate and prepare the model input.

    Args:
        x: The "raw" model input.

    Returns:
        The prepared input.

    Raises:
        ValueError: If the model input is empty.
    """
    from baybe.utils.torch import DTypeFloatTorch

    if len(x) == 0:
        raise ValueError("The model input must be non-empty.")
    return x.to(DTypeFloatTorch)


def _prepare_targets(y: Tensor) -> Tensor:
    """Validate and prepare the model targets.

    Args:
        y: The "raw" model targets.

    Returns:
        The prepared targets.

    Raises:
        NotImplementedError: If there is more than one target.
    """
    from baybe.utils.torch import DTypeFloatTorch

    if y.shape[1] != 1:
        raise NotImplementedError(
            "The model currently supports only one target or multiple targets in "
            "DESIRABILITY mode."
        )
    return y.to(DTypeFloatTorch)


def catch_constant_targets(cls: type[Surrogate], std_threshold: float = 1e-6):
    """Make a ``Surrogate`` class robustly handle constant training targets.

    More specifically, "constant training targets" can mean either of:
        * The standard deviation of the training targets is below the given threshold.
        * There is only one target and the standard deviation cannot even be computed.

    The modified class handles the above cases separately from "regular operation"
    by resorting to a :class:`baybe.surrogates.naive.MeanPredictionSurrogate`.

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

    def _posterior_new(self, candidates: Tensor) -> tuple[Tensor, Tensor]:
        # Alternative model fallback
        if isinstance(self._model, MeanPredictionSurrogate):
            return self._model._posterior(candidates)

        # Regular operation
        return _posterior_original(self, candidates)

    def _fit_new(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> None:
        if not (train_y.ndim == 2 and train_y.shape[-1] == 1):
            raise NotImplementedError(
                "The current logic is only implemented for single-target surrogates."
            )

        # Alternative model fallback
        if train_y.numel() == 1 or train_y.std() < std_threshold:
            self._model = MeanPredictionSurrogate()
            self._model._fit(searchspace, train_x, train_y)
            return

        # Regular operation
        _fit_original(self, searchspace, train_x, train_y)

    # Replace the methods
    cls._posterior = _posterior_new
    cls._fit = _fit_new

    return cls


def autoscale(cls: type[Surrogate]):
    """Make a ``Surrogate`` class automatically scale the domain it operates on.

    More specifically, the modified class transforms its inputs before operation and
    untransforms results before returning them.

    Args:
        cls: The :class:`baybe.surrogates.base.Surrogate` to be augmented.

    Returns:
        The modified class.
    """
    # References to original methods
    _fit_original = cls._fit
    _posterior_original = cls._posterior

    def _posterior_new(self, candidates: Tensor) -> tuple[Tensor, Tensor]:
        candidates = self._autoscaler.transform(candidates)
        mean, covar = _posterior_original(self, candidates)
        return self._autoscaler.untransform(mean, covar)

    def _fit_new(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> None:
        try:
            self._autoscaler = DefaultScaler(searchspace.discrete.comp_rep)
        except AttributeError as ex:
            raise TypeError(
                f"'{autoscale.__name__}' is only applicable to non-slotted classes "
                f"but '{cls.__name__}' is a slotted class."
            ) from ex
        train_x, train_y = self._autoscaler.fit_transform(train_x, train_y)
        _fit_original(self, searchspace, train_x, train_y)

    # Replace the methods
    cls._posterior = _posterior_new
    cls._fit = _fit_new

    return cls


def batchify(
    posterior: Callable[[Surrogate, Tensor], tuple[Tensor, Tensor]]
) -> Callable[[Surrogate, Tensor], tuple[Tensor, Tensor]]:
    """Wrap ``Surrogate`` posterior functions to enable proper batching.

    More precisely, this wraps model that are incompatible with t- and q-batching such
    that they become able to process batched inputs.

    Args:
        posterior: The original ``posterior`` function.

    Returns:
        The wrapped posterior function.
    """

    @wraps(posterior)
    def sequential_posterior(
        model: Surrogate, candidates: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Replace the posterior function by one that processes batches sequentially.

        Args:
            model: The ``Surrogate`` model.
            candidates: The candidates tensor.

        Returns:
            The mean and the covariance.
        """
        import torch

        # If no batch dimensions are given, call the model directly
        if candidates.ndim == 2:
            return posterior(model, candidates)

        # Keep track of batch dimensions
        t_shape = candidates.shape[:-2]
        q_shape = candidates.shape[-2]

        # If the posterior function provides full covariance information, call it
        # t-batch by t-batch
        if model.joint_posterior:
            # Flatten all t-batch dimensions into a single one
            flattened = candidates.flatten(end_dim=-3)

            # Call the model on each (flattened) t-batch
            out = (posterior(model, batch) for batch in flattened)

            # Collect the results and restore the batch dimensions
            mean, covar = zip(*out)
            mean = torch.reshape(torch.stack(mean), t_shape + (q_shape,))
            covar = torch.reshape(torch.stack(covar), t_shape + (q_shape, q_shape))

            return mean, covar

        # Otherwise, flatten all t- and q-batches into a single q-batch dimension
        # and evaluate the posterior function in one go
        else:
            # Flatten *all* batches into the q-batch dimension
            flattened = candidates.flatten(end_dim=-2)

            # Call the model on the entire input
            mean, var = posterior(model, flattened)

            # Restore the batch dimensions
            mean = torch.reshape(mean, t_shape + (q_shape,))
            var = torch.reshape(var, t_shape + (q_shape,))

            return mean, var

    return sequential_posterior
