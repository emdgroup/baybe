"""Target utilities."""

from __future__ import annotations

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
