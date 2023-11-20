"""Utilities for working with surrogates."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Callable, ClassVar, Tuple, Type

import torch
from torch import Tensor

from baybe.scaler import DefaultScaler
from baybe.searchspace import SearchSpace

if TYPE_CHECKING:
    from baybe.surrogates.base import Surrogate

# Use float64 (which is recommended at least for BoTorch models)
_DTYPE = torch.float64

_MIN_TARGET_STD = 1e-6


def _prepare_inputs(x: Tensor) -> Tensor:
    """Validate and prepare the model input.

    Args:
        x: The "raw" model input.

    Returns:
        The prepared input.

    Raises:
        ValueError: If the model input is empty.
    """
    if len(x) == 0:
        raise ValueError("The model input must be non-empty.")
    return x.to(_DTYPE)


def _prepare_targets(y: Tensor) -> Tensor:
    """Validate and prepare the model targets.

    Args:
        y: The "raw" model targets.

    Returns:
        The prepared targets.

    Raises:
        NotImplementedError: If there is more than one target.
    """
    if y.shape[1] != 1:
        raise NotImplementedError(
            "The model currently supports only one target or multiple targets in "
            "DESIRABILITY mode."
        )
    return y.to(_DTYPE)


def catch_constant_targets(model_cls: Type[Surrogate]):
    """Wrap a ``Surrogate`` class that cannot handle constant training target values.

    In the wrapped class, these cases are handled by a separate model type.

    Args:
        model_cls: A ``Surrogate`` class that should be wrapped.

    Returns:
        A wrapped version of the class.
    """

    class SplitModel(*model_cls.__bases__):
        """The class that is used for wrapping.

        It applies a separate strategy for cases where the training
        targets are all constant and no variance can be estimated.

        It stores an instance of the underlying model class.
        """

        # The posterior mode is chosen to match that of the wrapped model class
        joint_posterior: ClassVar[bool] = model_cls.joint_posterior
        # See base class.

        def __init__(self, *args, **kwargs):
            super().__init__()
            self.model = model_cls(*args, **kwargs)
            self.__class__.__name__ = self.model.__class__.__name__
            self.model_params = self.model.model_params

        def _posterior(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
            """Call the posterior function of the internal model instance."""
            mean, var = self.model._posterior(candidates)

            # If a joint posterior is expected but the model has been overridden by one
            # that does not provide covariance information, construct a diagonal
            # covariance matrix
            if self.joint_posterior and not self.model.joint_posterior:
                # Convert to tensor containing covariance matrices
                var = torch.diag_embed(var)

            return mean, var

        def _fit(
            self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
        ) -> None:
            """Select a model based on the variance of the targets and fits it."""
            from baybe.surrogates.naive import MeanPredictionSurrogate

            # https://github.com/pytorch/pytorch/issues/29372
            # Needs 'unbiased=False' (otherwise, the result will be NaN for scalars)
            if torch.std(train_y.ravel(), unbiased=False) < _MIN_TARGET_STD:
                self.model = MeanPredictionSurrogate()

            # Fit the selected model with the training data
            self.model.fit(searchspace, train_x, train_y)

        def __getattribute__(self, attr):
            """Access the attributes of the class instance if available.

            If the attributes are not available, it uses the attributes of the internal
            model instance.
            """
            # Try to retrieve the attribute in the class
            try:
                val = super().__getattribute__(attr)
            except AttributeError:
                pass
            else:
                return val

            # If the attribute has not been overwritten, use that of the internal model
            return self.model.__getattribute__(attr)

    # Wrapping a class using a decorator does not transfer the doc, resulting in the
    # autodocumentation not showing the correct docstring. We thus need to manually
    # assign the docstring of the class.
    SplitModel.__doc__ = model_cls.__doc__
    return SplitModel


def scale_model(model_cls: Type[Surrogate]):
    """Wrap a ``Surrogate`` class such that it operates with scaled representations.

    Args:
        model_cls: A ``Surrogate`` model class that should be wrapped.

    Returns:
        A wrapped version of the class.
    """

    class ScaledModel(*model_cls.__bases__):
        """Overrides the methods of the given model class such the use scaled data.

        It stores an instance of the underlying model class and a scalar object.
        """

        # The posterior mode is chosen to match that of the wrapped model class
        joint_posterior: ClassVar[bool] = model_cls.joint_posterior
        # See base class.

        def __init__(self, *args, **kwargs):
            self.model = model_cls(*args, **kwargs)
            self.__class__.__name__ = self.model.__class__.__name__
            self.model_params = self.model.model_params
            self.scaler = None

        def _posterior(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
            """Call the posterior function of the internal model instance.

            This call is made on a scaled version of the test data and rescales the
            output accordingly.
            """
            candidates = self.scaler.transform(candidates)
            mean, covar = self.model._posterior(candidates)
            return self.scaler.untransform(mean, covar)

        def _fit(
            self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
        ) -> None:
            """Fits the scaler and the model using the scaled training data."""
            self.scaler = DefaultScaler(searchspace.discrete.comp_rep)
            train_x, train_y = self.scaler.fit_transform(train_x, train_y)
            self.model.fit(searchspace, train_x, train_y)

        def __getattribute__(self, attr):
            """Access the attributes of the class instance if available.

            If the attributes are not available, it uses the attributes of the internal
            model instance.
            """
            # Try to retrieve the attribute in the class
            try:
                val = super().__getattribute__(attr)
            except AttributeError:
                pass
            else:
                return val

            # If the attribute has not been overwritten, use that of the internal model
            return self.model.__getattribute__(attr)

    # Wrapping a class using a decorator does not transfer the doc, resulting in the
    # autodocumentation not showing the correct docstring. We thus need to manually
    # assign the docstring of the class.
    ScaledModel.__doc__ = model_cls.__doc__
    return ScaledModel


def batchify(
    posterior: Callable[[Surrogate, Tensor], Tuple[Tensor, Tensor]]
) -> Callable[[Surrogate, Tensor], Tuple[Tensor, Tensor]]:
    """Wrap ``Surrogate`` posterior functions to enable proper batching.

    More precisely, this wraps model that are incompatible with t- and q-batching such
    that they become able to process batched inputs.

    Args:
        posterior: The original ``posterior`` function.

    Returns:
        The wrapped posterior function.
    """

    @wraps(posterior)
    def sequential_posterior(model: Surrogate, candidates: Tensor) -> [Tensor, Tensor]:
        """Replace the posterior function by one that processes batches sequentially.

        Args:
            model: The ``Surrogate`` model.
            candidates: The candidates tensor.

        Returns:
            The mean and the covariance.
        """
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
