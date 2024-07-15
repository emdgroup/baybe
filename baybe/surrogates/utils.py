"""Utilities for working with surrogates."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any

from baybe.scaler import DefaultScaler
from baybe.searchspace import SearchSpace

if TYPE_CHECKING:
    from botorch.posteriors import Posterior
    from torch import Tensor

    from baybe.surrogates.base import Surrogate
    from baybe.surrogates.naive import MeanPredictionSurrogate


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

    def _posterior_new(self, candidates: Tensor) -> Posterior:
        """Use fallback model if it exists, otherwise call original posterior."""
        # Alternative model fallback
        if constant_target_model := _constant_target_model_store.get(id(self), None):
            return constant_target_model._posterior(candidates)

        # Regular operation
        return _posterior_original(self, candidates)

    def _fit_new(self, train_x: Tensor, train_y: Tensor, context: Any) -> None:
        """Original fit but with fallback model creation for constant targets."""
        if not (train_y.ndim == 2 and train_y.shape[-1] == 1):
            raise NotImplementedError(
                "The current logic is only implemented for single-target surrogates."
            )

        # Alternative model fallback
        if train_y.numel() == 1 or train_y.std() < std_threshold:
            model = MeanPredictionSurrogate()
            model._fit(train_x, train_y, context)
            _constant_target_model_store[id(self)] = model

        # Regular operation
        else:
            _constant_target_model_store.pop(id(self), None)
            _fit_original(self, train_x, train_y, context)

    # Replace the methods
    cls._posterior = _posterior_new
    cls._fit = _fit_new

    return cls


def autoscale(cls: type[Surrogate]):
    """Make a ``Surrogate`` class automatically scale the domain it operates on.

    More specifically, the modified class transforms its inputs before processing them
    and untransforms the results before returning them. The fitted scaler used for these
    transformations is stored in the class' objects as an additional temporary
    attribute.

    Args:
        cls: The :class:`baybe.surrogates.base.Surrogate` to be augmented.

    Raises:
        ValueError: If the class already contains an attribute with the same name
            as the temporary attribute to be added.

    Returns:
        The modified class.
    """
    # Name of the attribute added to store the scaler
    injected_scaler_attr_name = "_autoscaler"

    if injected_scaler_attr_name in (attr.name for attr in cls.__attrs_attrs__):
        raise ValueError(
            f"Cannot apply '{autoscale.__name__}' because "
            f"'{cls.__name__}' already has an attribute '{injected_scaler_attr_name}' "
            f"defined."
        )

    # References to original methods
    _fit_original = cls._fit
    _posterior_original = cls._posterior

    def _posterior_new(self, candidates: Tensor) -> tuple[Tensor, Tensor]:
        scaled = getattr(self, injected_scaler_attr_name).transform(candidates)
        mean, covar = _posterior_original(self, scaled)
        return getattr(self, injected_scaler_attr_name).untransform(mean, covar)

    def _fit_new(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> None:
        scaler = DefaultScaler(searchspace.discrete.comp_rep)
        scaled_x, scaled_y = scaler.fit_transform(train_x, train_y)
        try:
            setattr(self, injected_scaler_attr_name, scaler)
        except AttributeError as ex:
            raise TypeError(
                f"'{autoscale.__name__}' is only applicable to "
                f"non-slotted classes but '{cls.__name__}' is a slotted class."
            ) from ex
        _fit_original(self, searchspace, scaled_x, scaled_y)

    # Replace the methods
    cls._posterior = _posterior_new
    cls._fit = _fit_new

    return cls


def batchify(
    posterior: Callable[[Surrogate, Tensor], tuple[Tensor, Tensor]],
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
