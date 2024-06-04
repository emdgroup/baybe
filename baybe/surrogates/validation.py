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


def get_model_params_validator(model_init: Callable | None = None) -> Callable:
    """Construct a validator based on the model class.

    Args:
        model_init: The init method for the model.

    Returns:
        A validator function to validate parameters.
    """

    def validate_model_params(  # noqa: DOC101, DOC103
        obj: Any, _: Any, model_params: dict
    ) -> None:
        """Validate the model params attribute of an object.

        Raises:
            ValueError: When model params are given for non-supported objects.
            ValueError: When surrogate is not recognized (no valid model_init).
            ValueError: When invalid params are given for a model.
        """
        # Get model class name
        model = obj.__class__.__name__

        if not model_params:
            return

        # GP does not support additional model params
        # Neither does custom models
        if "GaussianProcess" in model or "Custom" in model:
            raise ValueError(f"{model} does not support model params.")

        if not model_init:
            raise ValueError(
                f"Cannot validate model params for unrecognized Surrogate: {model}"
            )

        # Invalid params
        invalid_params = ", ".join(
            [
                key
                for key in model_params.keys()
                if key not in model_init.__code__.co_varnames
            ]
        )

        if invalid_params:
            raise ValueError(f"Invalid model params for {model}: {invalid_params}.")

    return validate_model_params
