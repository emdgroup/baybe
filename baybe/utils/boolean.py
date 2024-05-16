"""Functions implementing boolean checks."""

from abc import ABC
from typing import Any

from attrs import cmp_using
from typing_extensions import is_protocol

# Used for comparing pandas dataframes in attrs classes
eq_dataframe = cmp_using(lambda x, y: x.equals(y))


def is_abstract(cls: Any) -> bool:
    """Determine if a given class is abstract.

    This check is more general sense than ``inspect.abstract``, which only verifies
    if a class has abstract methods. The latter can be problematic when the class has
    no abstract methods but is nevertheless not directly usable, for example, because it
    has uninitialized members, which are only covered in its non-"abstract" subclasses.

    By contrast, this method simply checks if the class derives from ``abc.ABC`` or
    is a protocol class.

    Args:
        cls: The class to be inspected.

    Returns:
        ``True`` if the class is "abstract" (see definition above), ``False`` else.
    """
    return ABC in cls.__bases__ or is_protocol(cls)


def strtobool(val: str) -> bool:
    """Convert a string representation of truth to ``True`` or ``False``.

    Adapted from distutils.
    True values are ``y``, ``yes``, ``t``, ``true``, ``on``, and ``1``.
    False values are ``n``, ``no``, ``f``, ``false``, ``off``, and ``0``.
    Raises a ``ValueError`` if ``val`` is anything else.

    Args:
        val: String to be checked.

    Returns:
        The ``bool`` value of the corresponding string representation.

    Raises:
        ValueError: If ``val`` cannot be evaluated to a suitable boolean value.
    """
    if val.lower() in ("y", "yes", "t", "true", "on", "1"):
        return True
    if val.lower() in ("n", "no", "f", "false", "off", "0"):
        return False

    raise ValueError(f"Invalid truth value: {val}")


def check_if_in(element: Any, allowed: list):
    """Check if an element is in a given list of elements.

    Args:
        element: The element to be checked
        allowed: The corresponding list

    Raises:
        ValueError: If ``element`` is not in ``allowed``.
    """
    if element not in allowed:
        raise ValueError(
            f"The value '{element}' is not allowed. Must be one of {allowed}."
        )
