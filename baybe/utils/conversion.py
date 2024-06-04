"""Conversion utilities."""

from collections.abc import Sequence
from fractions import Fraction
from typing import TypeVar

from attrs import Attribute

_T = TypeVar("_T")


def fraction_to_float(value: str | float | Fraction, /) -> float:
    """Convert the provided input representing a fraction into a float.

    Args:
        value: The input to be converted.

    Returns:
        The float representation of the given input.

    Raises:
        ValueError: If the input was provided as string but could not be interpreted as
            fraction.
    """
    if isinstance(value, str):
        try:
            value = Fraction(value)
        except ValueError as err:
            raise ValueError(
                f"The provided input '{value}' could not be interpreted as a fraction."
            ) from err
    return float(value)


def nonstring_to_tuple(x: Sequence[_T], self: type, field: Attribute) -> tuple[_T, ...]:
    """Convert a sequence to tuple but raise an exception for string input."""
    if isinstance(x, str):
        raise ValueError(
            f"Argument passed to '{field.alias}' of class '{self.__class__.__name__}' "
            f"must be a sequence but cannot be a string."
        )
    return tuple(x)
