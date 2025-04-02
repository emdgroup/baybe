"""Plotting utilities."""

from typing import Any


def indent(text: str, amount: int = 3, ch: str = " ") -> str:
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

    return "\n".join([header] + [indent(str(f)) for f in fields])
