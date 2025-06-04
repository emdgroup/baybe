"""Validation functionality for surrogates."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cattrs
from cattrs import ClassValidationError
from cattrs.strategies import configure_union_passthrough

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


# Create a strict type validation converter
type_validation_converter = cattrs.Converter(forbid_extra_keys=True)
"""Converter used for strict type validation."""

configure_union_passthrough(int | float | str | None, type_validation_converter)


@type_validation_converter.register_structure_hook
def _strict_int_structure_hook(obj: Any, _: type[int]) -> int:
    if isinstance(obj, int) and not isinstance(obj, bool):  # Exclude bools
        return obj
    raise ValueError(
        f"Value '{obj}' (type: {type(obj).__name__}) is not a valid integer. "
        "Only actual 'int' instances are accepted."
    )


@type_validation_converter.register_structure_hook
def _strict_float_structure_hook(obj: Any, _: type[float]) -> float:
    if isinstance(obj, float):
        return obj
    raise ValueError(
        f"Value '{obj}' (type: {type(obj).__name__}) is not a valid float. "
        "Only actual 'float' instances are accepted."
    )


@type_validation_converter.register_structure_hook
def _strict_bool_structure_hook(obj: Any, _: type[bool]) -> bool:
    if isinstance(obj, bool):
        return obj
    raise ValueError(
        f"Value '{obj}' (type: {type(obj).__name__}) is not a valid boolean. "
        "Only actual 'bool' instances (True, False) are accepted."
    )


def make_dict_validator(specification: type) -> Callable:
    """Construct an attrs dictionary validator based on a ``TypedDict``.

    Args:
        specification: Describes allowed keys and corresponding value types.

    Returns:
        An attrs compatible validator.
    """

    def validate_model_params(_instance: Any, attr: Any, value: dict) -> None:
        """Validate attrs attribute using cattrs with an extremely strict int hook."""
        try:
            type_validation_converter.structure(value, specification)
        except ClassValidationError as ex:
            raise TypeError(
                f"The provided dictionary for '{attr.name}' is invalid."
            ) from ex

    return validate_model_params
