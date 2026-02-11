"""Converter and hooks."""

from __future__ import annotations

import base64
import pickle
from datetime import datetime, timedelta
from typing import (
    TYPE_CHECKING,
    Any,
    NoReturn,
    TypeVar,
    get_args,
    get_type_hints,
)

import attrs
import cattrs
import pandas as pd
from cattrs.strategies import configure_union_passthrough

from baybe.utils.basic import find_subclass
from baybe.utils.boolean import is_abstract

if TYPE_CHECKING:
    from cattrs.dispatch import UnstructureHook

_T = TypeVar("_T")

_TYPE_FIELD = "type"
"""The name of the field used to store the type information in serialized objects."""

converter = cattrs.Converter(unstruct_collection_overrides={set: list}, use_alias=True)
"""The default converter for (de-)serializing BayBE-related objects."""


def _add_type_to_dict(dct: dict[str, Any], type_: str, /) -> dict[str, Any]:
    """Safely add type information to an existing dictionary."""
    if _TYPE_FIELD in dct:
        raise ValueError(
            f"Cannot add type information to the dictionary since it already contains "
            f"a '{_TYPE_FIELD}' field."
        )
    dct = {_TYPE_FIELD: type_, **dct}
    return dct


def add_type(hook: UnstructureHook) -> UnstructureHook:
    """Wrap a given hook to add type information to the unstructured object."""

    def wrapper(obj: Any, /) -> dict[str, Any]:
        """Unstructure an object and add its type information."""
        dct = hook(obj)
        return _add_type_to_dict(dct, obj.__class__.__name__)

    return wrapper


def unstructure_with_type(x: Any, /) -> dict[str, Any]:
    """Unstructure an object and add its type information."""
    return add_type(converter.get_unstructure_hook(x.__class__))(x)


def make_base_unstructure_hook(base: type[_T]):
    """Create a hook for unstructuring subclasses using annotations of their base class.

    For generic base classes, the created hook preserves the generic type parameters
    when unstructuring fields. For example, if a field is annotated as a parameterized
    generic (e.g., ``BaseClass[Type]``), the field value will be unstructured using
    that parameterized type rather than just the runtime concrete subclass.
    """
    if not is_abstract(base):
        raise ValueError(
            f"Registering base class hooks is intended for abstract classes "
            f"only. Given: '{base.__name__}' (which is not abstract).",
        )

    def unstructure_base(obj: Any) -> dict[str, Any]:
        if type_args := get_args(base):
            # For parameterized generics (e.g., BaseClass[Type]), we use its type
            # information when unstructuring, not just the plain runtime class
            concrete_cls = obj.__class__
            try:
                parameterized_cls = concrete_cls[type_args]
                hook = converter.get_unstructure_hook(parameterized_cls)
            except TypeError:
                # Non-generic classes do not support parameterization
                hook = converter.get_unstructure_hook(concrete_cls)
        else:
            hook = converter.get_unstructure_hook(obj.__class__)

        dct = hook(obj)
        return _add_type_to_dict(dct, obj.__class__.__name__)

    return unstructure_base


def make_base_structure_hook(base: type[_T]):
    """Create a hook for structuring subclasses using annotations of their base class.

    Reads the ``type`` information from the given input to retrieve the correct
    subclass and then calls the existing structure hook of the that class.

    For generic base classes, this hook preserves the generic type parameters when
    dispatching to concrete subclasses. For example, if structuring with
    ``BaseClass[Type]``, the concrete subclass will be called as ``ConcreteClass[Type]``
    rather than just the plain ``ConcreteClass``.
    """
    if not is_abstract(base):
        raise ValueError(
            f"Registering base class hooks is intended for abstract classes "
            f"only. Given: '{base.__name__}' (which is not abstract).",
        )

    def structure_base(val: dict[str, Any] | str, cls: type[_T]) -> _T:
        # Extract the type information from the given input and find
        # the corresponding class in the hierarchy
        type_ = val if isinstance(val, str) else val.pop(_TYPE_FIELD)
        subclass = find_subclass(base, type_)

        # Preserve generic type parameters from the abstract base class:
        # If class is a parameterized generic (e.g., BaseClass[Type]),
        # apply the same parameters to subclass (e.g., SubClass[Kernel])
        if type_args := get_args(cls):
            subclass = subclass[type_args]

        # Call the structure hook of the concrete class
        fn = converter.get_structure_hook(subclass)
        return fn({} if isinstance(val, str) else val, subclass)

    return structure_base


def _structure_dataframe_hook(obj: str | dict, _) -> pd.DataFrame:
    """Deserialize a DataFrame."""
    if isinstance(obj, str):
        pickled_df = base64.b64decode(obj.encode("utf-8"))
        return pickle.loads(pickled_df)
    elif isinstance(obj, dict):
        if "constructor" not in obj:
            raise ValueError(
                "For deserializing a dataframe from a dictionary, the 'constructor' "
                "keyword must be provided as key.",
            )
        return select_constructor_hook(obj, pd.DataFrame)
    else:
        raise ValueError(
            "Unknown object type for deserializing a dataframe. Supported types are "
            "strings and dictionaries.",
        )


def _unstructure_dataframe_hook(df: pd.DataFrame) -> str:
    """Serialize a DataFrame."""
    pickled_df = pickle.dumps(df)
    return base64.b64encode(pickled_df).decode("utf-8")


def block_serialization_hook(obj: Any) -> NoReturn:  # noqa: DOC101, DOC103
    """Prevent serialization of the passed object.

    Raises:
        NotImplementedError: Always.
    """
    raise NotImplementedError(
        f"Serializing objects of type '{obj.__class__.__name__}' is not supported."
    )


def block_deserialization_hook(_: Any, cls: type) -> NoReturn:  # noqa: DOC101, DOC103
    """Prevent deserialization into a specific type.

    Raises:
        NotImplementedError: Always.
    """
    raise NotImplementedError(
        f"Deserialization into '{cls.__name__}' is not supported."
    )


def select_constructor_hook(specs: dict, cls: type[_T]) -> _T:
    """Use the constructor specified in the 'constructor' field for deserialization."""
    # If a constructor is specified, use it
    specs = specs.copy()
    if constructor_name := specs.pop("constructor", None):
        constructor = getattr(cls, constructor_name)

        # If given a non-attrs class, simply call the constructor
        if not attrs.has(cls):
            return constructor(**specs)

        # Extract the constructor parameter types and deserialize the arguments
        type_hints = get_type_hints(constructor)
        for key, value in specs.items():
            annotation = type_hints[key]
            specs[key] = converter.structure(specs[key], annotation)

        # Call the constructor with the deserialized arguments
        return constructor(**specs)

    # Otherwise, use the regular __init__ method
    return converter.structure_attrs_fromdict(specs, cls)


# Register custom (un-)structure hooks
configure_union_passthrough(bool | int | float | str, converter)
converter.register_unstructure_hook_factory(
    lambda cls: is_abstract(cls) and cls.__module__.startswith("baybe."),
    make_base_unstructure_hook,
)

converter.register_structure_hook_factory(
    lambda cls: is_abstract(cls) and cls.__module__.startswith("baybe."),
    make_base_structure_hook,
)
converter.register_unstructure_hook(pd.DataFrame, _unstructure_dataframe_hook)
converter.register_structure_hook(pd.DataFrame, _structure_dataframe_hook)
converter.register_unstructure_hook(datetime, lambda x: x.isoformat())
converter.register_structure_hook(datetime, lambda x, _: datetime.fromisoformat(x))
converter.register_unstructure_hook(timedelta, lambda x: f"{x.total_seconds()}s")
converter.register_structure_hook(
    timedelta, lambda x, _: timedelta(seconds=float(x.removesuffix("s")))
)
