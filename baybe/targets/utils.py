"""Target utilities."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, Concatenate, ParamSpec, TypeVar

from attrs import evolve, fields, fields_dict

from baybe.transformations.basic import IdentityTransformation

_T = TypeVar("_T")

if TYPE_CHECKING:
    from baybe.targets.numerical import NumericalTarget

    P = ParamSpec("P")


def _validate_numerical_target_combination(
    t1: NumericalTarget, t2: NumericalTarget, /
) -> None:
    """Validate if two numerical targets can be combined."""
    from baybe.targets.numerical import NumericalTarget

    t1_ = evolve(t1, transformation=IdentityTransformation())  # type: ignore[call-arg]
    t2_ = evolve(t2, transformation=IdentityTransformation())  # type: ignore[call-arg]
    if t1_ != t2_:
        raise ValueError(
            f"Two objects of type '{NumericalTarget.__name__}' can only be "
            f"combined if they are identical in all attributes except for the "
            f"'{fields(NumericalTarget).transformation.name}'. "
            f"Given: {t1_!r} and {t2_!r}."
        )


def combine_numerical_targets(
    t1: NumericalTarget, t2: NumericalTarget, /, operator
) -> NumericalTarget:
    """Combine two numerical targets using a binary operator."""
    _validate_numerical_target_combination(t1, t2)
    return evolve(t1, transformation=operator(t1.transformation, t2.transformation))  # type: ignore[call-arg]


def capture_constructor_info(
    constructor: Callable[Concatenate[type[_T], P], _T],
) -> Callable[Concatenate[type[_T], P], _T]:
    """Capture constructor history upon object creation.

    To be used as decorator with classmethods.
    """

    @wraps(constructor)
    def wrapper(cls: type[_T], *args: P.args, **kwargs: P.kwargs) -> _T:
        obj = constructor(cls, *args, **kwargs)

        # Reconstruct arguments
        sig = inspect.signature(constructor)
        bound = sig.bind(cls, *args, **kwargs)
        bound.apply_defaults()  # To make it consistent with results for __init__
        bound.arguments.pop("cls")  # Ignore "cls"

        # Store argument history
        constructor_info: dict[str, Any] = {
            "constructor": constructor.__name__,
            **{
                k: v
                for k, v in bound.arguments.items()
                if k not in fields_dict(obj.__class__)  # Ignore persistent attributes
            },
        }
        object.__setattr__(
            obj, fields(obj.__class__)._constructor_info.name, constructor_info
        )

        return obj

    return wrapper
