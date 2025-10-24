"""Functions implementing Boolean checks."""

from __future__ import annotations

import enum
from abc import ABC
from collections.abc import Callable
from typing import Any

from attrs import cmp_using
from typing_extensions import is_protocol, override

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
        ValueError: If ``val`` cannot be evaluated to a suitable Boolean value.
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


class UncertainBool(enum.Enum):
    """Enum for representing uncertain Boolean values."""

    TRUE = "TRUE"
    """Represents the Boolean value `True`."""

    FALSE = "FALSE"
    """Represents the Boolean value `False`."""

    UNKNOWN = "UNKNOWN"
    """Indicates that the value of the Boolean cannot be determined."""

    def __bool__(self):
        if self is UncertainBool.TRUE:
            return True
        elif self is UncertainBool.FALSE:
            return False
        elif self is UncertainBool.UNKNOWN:
            raise TypeError(f"'{UncertainBool.UNKNOWN}' has no Boolean representation.")
        raise ValueError(f"Unknown value: '{self}'")

    @classmethod
    def from_erroneous_callable(cls, callable_: Callable, /) -> UncertainBool:
        """Create an uncertain Boolean from a potentially erroneous Boolean call."""
        try:
            return cls.TRUE if callable_() else cls.FALSE
        except Exception:
            return cls.UNKNOWN


class AutoBool(enum.Enum):
    """Enum for representing Booleans whose values can be determined automatically."""

    # https://github.com/python-attrs/attrs/issues/1462
    __hash__ = object.__hash__

    TRUE = "TRUE"
    """Represents the Boolean value `True`."""

    FALSE = "FALSE"
    """Represents the Boolean value `False`."""

    AUTO = "AUTO"
    """
    Indicates that the value of the Boolean should be determined automatically
    on-the-fly, using a predicate function.
    """

    def __bool__(self):
        if self is AutoBool.TRUE:
            return True
        elif self is AutoBool.FALSE:
            return False
        elif self is AutoBool.AUTO:
            raise TypeError(f"'{AutoBool.AUTO}' has no Boolean representation.")
        raise ValueError(f"Unknown value: '{self}'")

    @override
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, (bool, AutoBool)):
            raise NotImplementedError

        if self is AutoBool.TRUE or self is AutoBool.FALSE:
            return bool(self) == other
        elif isinstance(other, AutoBool) and other is AutoBool.AUTO:
            return True
        return False

    def evaluate(self, predicate_function: Callable[[], bool]) -> bool:
        """Evaluate the Boolean value under a given predicate function."""
        if self is AutoBool.TRUE:
            return True
        elif self is AutoBool.FALSE:
            return False
        elif self is AutoBool.AUTO:
            return predicate_function()
        raise ValueError(f"Unknown value: '{self}'")

    @classmethod
    def from_unstructured(cls, value: AutoBool | bool | str | None, /) -> AutoBool:
        """Create the enum member from unstructured input.

        For string inputs, see :func:`~baybe.utils.boolean.strtobool`.

        Args:
            value: The (possibly unstructured) input value to be converted.

        Returns:
            The corresponding enum member.

        Raises:
            ValueError: If the input cannot be converted to an enum member.

        Example:
            >>> AutoBool.from_unstructured(AutoBool.TRUE)
            <AutoBool.TRUE: 'TRUE'>

            >>> AutoBool.from_unstructured(True)
            <AutoBool.TRUE: 'TRUE'>

            >>> AutoBool.from_unstructured("t")
            <AutoBool.TRUE: 'TRUE'>

            >>> AutoBool.from_unstructured(AutoBool.FALSE)
            <AutoBool.FALSE: 'FALSE'>

            >>> AutoBool.from_unstructured(False)
            <AutoBool.FALSE: 'FALSE'>

            >>> AutoBool.from_unstructured("f")
            <AutoBool.FALSE: 'FALSE'>

            >>> AutoBool.from_unstructured(AutoBool.AUTO)
            <AutoBool.AUTO: 'AUTO'>

            >>> AutoBool.from_unstructured(None)
            <AutoBool.AUTO: 'AUTO'>

            >>> AutoBool.from_unstructured("auto")
            <AutoBool.AUTO: 'AUTO'>
        """
        match value:
            case AutoBool():
                return value
            case bool() as b:
                return cls.TRUE if b else cls.FALSE
            case None:
                return cls.AUTO
            case str() as s:
                if s.lower() == "auto":
                    return cls.AUTO
                try:
                    return cls.from_unstructured(strtobool(s))
                except ValueError:
                    pass

        raise ValueError(f"Cannot convert '{value}' to '{cls.__name__}'.")
