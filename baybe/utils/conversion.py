"""Conversion utilities."""

from fractions import Fraction
from typing import Union


def fraction_to_float(value: Union[str, float, Fraction], /) -> float:
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
