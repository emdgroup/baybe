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


def nonstring_to_tuple(
    x: Sequence[_T],
    self: object = None,
    field: Attribute | None = None,
    /,
) -> tuple[_T, ...]:
    """Convert a sequence to tuple but raise an exception for string input.

    Can be used for plain conversion or as a converter for an attrs field.

    Args:
        x: The sequence to be converted.
        self: The object owning the field, used for error reporting. When provided,
            its class name is included in the error message.
        field: The field descriptor, used for error reporting. When provided, its
            alias is included in the error message.

    Returns:
        The tuple representation of the given sequence.

    Raises:
        ValueError: If the provided value is a string.
    """
    if isinstance(x, str):
        context = (
            "Argument"
            + (f" '{field.alias}'" if field is not None else "")
            + (f" of class '{self.__class__.__name__}'" if self is not None else "")
        )
        raise ValueError(f"{context} must be a sequence but cannot be a string.")
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
