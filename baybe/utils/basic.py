"""Collection of small basic utilities."""

import functools
import inspect
from collections.abc import Callable, Collection, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from attrs import asdict, has

from baybe.exceptions import UnidentifiedSubclassError, UnmatchedAttributeError

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
        {1: ['A', 'C']}
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


def match_attributes(
    object: Any,
    callable_: Callable,
    /,
    strict: bool = True,
    ignore: Collection[str] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Find the attributes of an object that match with a given callable signature.

    Parameters appearing in the callable signature that have no match with the given
    object attributes are ignored.

    Args:
        object: The object whose attributes are to be matched.
        callable_: The callable against whose signature the attributes are to be
            matched.
        strict: If ``True``, an error is raised when the object has attributes that
            are not found in the callable signature (see also ``ignore``).
            If ``False``, these attributes are returned separately.
        ignore: A collection of attributes names that are to be ignored during matching.

    Raises:
        ValueError: If applied to a non-attrs object.
        UnmatchedAttributeError: In strict mode, if not all attributes can be matched.

    Returns:
        * A dictionary mapping the matched attribute names to their values.
        * A dictionary mapping the unmatched attribute names to their values.
    """
    if not has(object.__class__):
        raise ValueError(
            f"'{match_attributes.__name__}' only works with attrs objects."
        )
    # Get attribute/parameter sets
    set_object = set(asdict(object)) - set(ignore)
    set_callable = set(inspect.signature(callable_).parameters)

    # Match
    in_signature = set_object.intersection(set_callable)
    not_in_signature = set_object - set_callable
    if strict and not_in_signature:
        raise UnmatchedAttributeError(
            f"The following attributes cannot be matched: {not_in_signature}."
        )

    # Collect attributes for both sets
    attrs_in_signature = {p: getattr(object, p) for p in in_signature}
    attrs_not_in_signature = {p: getattr(object, p) for p in not_in_signature}

    return attrs_in_signature, attrs_not_in_signature


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


def register_hooks(
    target: Callable,
    pre_hooks: Sequence[Callable] | None = None,
    post_hooks: Sequence[Callable] | None = None,
) -> Callable:
    """Register custom hooks with a given target callable.

    The provided hooks need to be "compatible" with the target in the sense that their
    signatures can be aligned:

    * The hook signature may only contain parameters that also exist in the target
      callable (<-- basic requirement).
    * However, parameters that are not needed by the hook can be omitted from
      its signature. This requires that the parameters of both signatures can be matched
      via their names. For simplicity, it is thus assumed that the hook has no
      positional-only arguments.
    * If an annotation is provided for a hook parameter, it must match its
      target counterpart (<-- safety mechanism to prevent unintended argument use).
      An exception is when the target parameter has no annotation, in which case
      the hook annotation is unrestricted. This is particularly useful when registering
      hooks with methods, since it offers the possibility to annotate the "self"
      parameter bound to the method-carrying object, which is typically not annotated
      in the target callable.

    Args:
        target: The callable to which the hooks are to be attached.
        pre_hooks: Hooks to be executed before calling the target.
        post_hooks: Hooks to be executed after calling the target.

    Returns:
        The wrapped callable with the hooks attached.

    Raises:
        TypeError: If any hook has positional-only arguments.
        TypeError: If any hook expects parameters that are not present in the target.
        TypeError: If any hook has a non-empty parameter annotation that does not
            match with the corresponding annotation of the target.
    """
    # Defaults
    pre_hooks = pre_hooks or []
    post_hooks = post_hooks or []

    target_signature = inspect.signature(target, eval_str=True).parameters

    # Validate hook signatures
    for hook in [*pre_hooks, *post_hooks]:
        hook_signature = inspect.signature(hook, eval_str=True).parameters

        if any(
            p.kind is inspect.Parameter.POSITIONAL_ONLY for p in hook_signature.values()
        ):
            raise TypeError("The provided hooks cannot have positional-only arguments.")

        if unrecognized := (set(hook_signature) - set(target_signature)):
            raise TypeError(
                f"The parameters expected by the hook '{hook.__name__}' must be a "
                f"subset of the parameter of the target callable '{target.__name__}'. "
                f"Unrecognized hook parameters: {unrecognized}."
            )

        for name, hook_param in hook_signature.items():
            target_param = target_signature[name]

            # If target parameter is not annotated, the hook annotation is unrestricted
            if (t_hint := target_param.annotation) is inspect.Parameter.empty:
                continue

            # If target parameter is annotated, the hook annotation must be compatible,
            # i.e., be identical or empty
            if ((h_hint := hook_param.annotation) != t_hint) and (
                h_hint is not inspect.Parameter.empty
            ):
                raise TypeError(
                    f"The type annotation for '{name}' is not consistent between "
                    f"the given hook '{hook.__name__}' and the target callable "
                    f"'{target.__name__}'. Given: {h_hint}. Expected: {t_hint}."
                )

    def pass_args(hook: Callable, *args, **kwargs) -> None:
        """Call the hook with its requested subset of arguments."""
        hook_signature = inspect.signature(hook, eval_str=True).parameters
        matched_args = dict(zip(target_signature, args))
        matched_kwargs = {
            p: kwargs.get(p, target_signature[p].default)
            for p in hook_signature
            if p not in matched_args
        }
        passed_kwargs = {p: (matched_args | matched_kwargs)[p] for p in hook_signature}
        hook(**passed_kwargs)

    @functools.wraps(target)
    def wraps(*args, **kwargs):
        for hook in pre_hooks:
            pass_args(hook, *args, **kwargs)
        result = target(*args, **kwargs)
        for hook in post_hooks:
            pass_args(hook, *args, **kwargs)
        return result

    return wraps
