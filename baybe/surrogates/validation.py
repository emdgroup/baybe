"""Validation functionality for surrogates."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from baybe.surrogates.base import Surrogate


def validate_custom_architecture_cls(model_cls: type) -> None:
    """Validate a custom architecture to have the correct attributes.

    Args:
        model_cls: The user defined model class.

    Raises:
        ValueError: When model_cls does not have _fit or _posterior.
        ValueError: When _fit or _posterior is not a callable method.
        ValueError: When _fit does not have the required signature.
        ValueError: When _posterior does not have the required signature.
    """
    # Methods must exist
    if not (hasattr(model_cls, "_fit") and hasattr(model_cls, "_posterior")):
        raise ValueError(
            "`_fit` and a `_posterior` must exist for custom architectures"
        )

    fit = model_cls._fit
    posterior = model_cls._posterior

    # They must be methods
    if not (callable(fit) and callable(posterior)):
        raise ValueError(
            "`_fit` and a `_posterior` must be methods for custom architectures"
        )

    # Methods must have the correct arguments
    params = fit.__code__.co_varnames[: fit.__code__.co_argcount]

    if params != Surrogate._fit.__code__.co_varnames:
        raise ValueError(
            "Invalid args in `_fit` method definition for custom architecture. "
            "Please refer to Surrogate._fit for the required function signature."
        )

    params = posterior.__code__.co_varnames[: posterior.__code__.co_argcount]

    if params != Surrogate._posterior.__code__.co_varnames:
        raise ValueError(
            "Invalid args in `_posterior` method definition for custom architecture. "
            "Please refer to Surrogate._posterior for the required function signature."
        )


def get_model_params_validator(model_params: dict) -> Callable:
    """Construct a validator based on the model type dict.

    Args:
        model_params: The init method for the model.

    Returns:
        A validator function to validate parameters.
    """

    def validate_model_params(  # noqa: DOC101, DOC103
        obj: Any, _: Any, model_params: dict
    ) -> None:
        """Validate the model params attribute of an object."""
        pass

    return validate_model_params
