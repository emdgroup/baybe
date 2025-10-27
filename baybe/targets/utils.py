"""Target utilities."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING

from attrs import evolve, fields

from baybe.transformations.basic import IdentityTransformation

if TYPE_CHECKING:
    from baybe.targets.numerical import NumericalTarget


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


def capture_constructor_metadata(
    constructor: Callable[..., NumericalTarget], /
) -> Callable[..., NumericalTarget]:
    """Decorator to capture constructor metadata upon object creation."""  # noqa: D401
    from baybe.targets.numerical import ConstructorMetadata, OptionalAttributes

    @wraps(constructor)
    def wrapper(*args: object, **kwargs: object) -> NumericalTarget:
        sig = inspect.signature(constructor)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        # We don't need to store the first argument, since it's the class itself
        bound.arguments.pop(next(iter(bound.arguments)))

        target = constructor(*args, **kwargs)
        object.__setattr__(
            target,
            "optional",
            OptionalAttributes(
                ConstructorMetadata(constructor.__name__, bound.arguments)
            ),
        )
        return target

    return wrapper
