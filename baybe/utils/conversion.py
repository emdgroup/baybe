"""Conversion utilities."""

from collections.abc import Sequence
from fractions import Fraction
from typing import Any, TypeVar

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


def _indent(text: str, amount: int = 3, ch: str = " ") -> str:
    """Indent a given text by a certain amount."""
    padding = amount * ch
    return "".join(padding + line for line in text.splitlines(keepends=True))


def to_string(header: str, *fields: Any, single_line: bool = False) -> str:
    """Create a nested string representation.

    Args:
        header: The header, typically the name of a class.
        *fields: Fields to be printed with an indentation.
        single_line: If ``True``, print the representation on a single line.
            Only applicable when given a single field.

    Raises:
        ValueError: If ``single_line`` is ``True`` but ``fields`` contains more than one
            element.

    Returns:
        The string representation with indented fields.
    """
    if single_line:
        if len(fields) > 1:
            raise ValueError(
                "``single_line`` is only applicable when given a single field."
            )
        # Since single line headers look ugly without a ":", we add it manually
        header = header if header.endswith(":") else header + ":"
        return f"{header} {str(fields[0])}"

    return "\n".join([header] + [_indent(str(f)) for f in fields])
