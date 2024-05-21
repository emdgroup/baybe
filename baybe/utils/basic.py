"""Collection of small basic utilities."""

from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from inspect import signature
from typing import Any, Callable, TypeVar

from baybe.exceptions import UnidentifiedSubclassError

_C = TypeVar("_C", bound=type)
_T = TypeVar("_T")
_U = TypeVar("_U")


@dataclass(frozen=True, repr=False)
class Dummy:
    """Placeholder element for array-like data types.

    Useful e.g. for detecting duplicates in constraints.
    """

    def __repr__(self):
        """Return a representation of the placeholder."""
        return "<dummy>"


def get_subclasses(cls: _C, recursive: bool = True, abstract: bool = False) -> list[_C]:
    """Return a list of subclasses for the given class.

    Args:
        cls: The base class to retrieve subclasses for.
        recursive: If ``True``, indirect subclasses (i.e. subclasses of subclasses)
            are included.
        abstract: If ``True``, abstract subclasses are included.

    Returns:
        A list of subclasses for the given class.
    """
    from baybe.utils.boolean import is_abstract

    subclasses = []
    for subclass in cls.__subclasses__():
        # Append direct subclass only if it is not abstract
        if abstract or not is_abstract(subclass):
            subclasses.append(subclass)

        # If requested, add indirect subclasses
        if recursive:
            subclasses.extend(get_subclasses(subclass, abstract=abstract))

    return subclasses


def get_baseclasses(
    cls: type,
    recursive: bool = True,
    abstract: bool = False,
) -> list[type]:
    """Return a list of base classes for the given class.

    Args:
        cls: The class to retrieve base classes for.
        recursive: If ``True``, indirect base classes (i.e., base classes of base
            classes) are included.
        abstract: If `True`, abstract base classes are included.

    Returns:
        A list of base classes for the given class.
    """
    from baybe.utils.boolean import is_abstract

    classes = []

    for baseclass in cls.__bases__:
        if baseclass not in classes:
            if abstract or not is_abstract(baseclass):
                classes.append(baseclass)

            if recursive:
                classes.extend(get_baseclasses(baseclass, abstract=abstract))

    return classes


def hilberts_factory(factory: Callable[..., _T]) -> Iterable[_T]:
    """Provide an infinite stream of the factory's products."""
    while True:
        yield factory()


def group_duplicate_values(dictionary: dict[_T, _U]) -> dict[_U, list[_T]]:
    """Identify groups of keys that have the same value.

    Args:
        dictionary: The dictionary to screen for duplicate values.

    Returns:
        A dictionary whose keys are a subset of values of the input dictionary,
        and whose values are lists that group original keys holding the same value.

    Example:
        >>> group_duplicate_values({"A": 1, "B": 2, "C": 1, "D": 3})
        {1: ['B', 'C']}
    """
    group: dict[_U, list[_T]] = {}
    for key, value in dictionary.items():
        group.setdefault(value, []).append(key)
    return {k: v for k, v in group.items() if len(v) > 1}


def to_tuple(x: Sequence) -> tuple:
    """Convert any sequence into a tuple.

    This wrapper is only used to avoid mypy warnings for attrs converters:
    * https://github.com/python/mypy/issues/8417
    * https://github.com/python/mypy/issues/8389
    * https://github.com/python/mypy/issues/5313
    """
    # TODO: Remove wrapper once mypy support is there
    return tuple(x)


def filter_attributes(
    object: Any,
    callable_: Callable,
    ignore: Collection[str] = ("self", "kwargs", "args"),
) -> dict[str, Any]:
    """Find the attributes of an object that match with a given callable signature.

    Parameters appearing in the callable signature that have no match with the given
    object attributes are ignored.

    Args:
        object: The object whose attributes are to be returned.
        callable_: The callable against whose signature the attributes are to be
            matched.
        ignore: A collection of parameter names to be ignored in the signature.

    Returns:
        A dictionary mapping the matched attribute names to their values.
    """
    params = signature(callable_).parameters
    return {
        p: getattr(object, p)
        for p in params
        if (p not in ignore) and hasattr(object, p)
    }


class classproperty:
    """A decorator to make class properties.

    A class property combines the characteristics of @property and @classmethod. The
    simple chaining of these two existing decorators is deprecated
    (https://docs.python.org/3.11/whatsnew/3.11.html#language-builtins) and causes mypy
    issues.
    """

    def __init__(self, fn: Callable) -> None:
        self.fn = fn

    def __get__(self, _, cl: type):
        return self.fn(cl)


def refers_to(cls: type, name_or_abbr: str, /) -> bool:
    """Check if the given name or abbreviation refers to the specified class."""
    return name_or_abbr in (
        (cls.__name__, cls.abbreviation)
        if hasattr(cls, "abbreviation")
        else (cls.__name__,)
    )


def find_subclass(base: type, name_or_abbr: str, /):
    """Retrieve a specific subclass of a base class via its name or abbreviation."""
    try:
        return next(cl for cl in get_subclasses(base) if refers_to(cl, name_or_abbr))
    except StopIteration:
        raise UnidentifiedSubclassError(
            f"The class name or abbreviation '{name_or_abbr}' does not refer to any "
            f"of the subclasses of '{base.__name__}'."
        )
