"""Converter and hooks."""

from __future__ import annotations

import base64
import pickle
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, TypeVar, get_type_hints

import attrs
import cattrs
import pandas as pd
from cattrs.strategies import configure_union_passthrough

from baybe.utils.basic import find_subclass, refers_to
from baybe.utils.boolean import is_abstract

if TYPE_CHECKING:
    from cattrs.dispatch import UnstructureHook

_T = TypeVar("_T")

_TYPE_FIELD = "type"
"""The name of the field used to store the type information in serialized objects."""

# TODO: This urgently needs the `forbid_extra_keys=True` flag, which requires us to
#   switch to the cattrs built-in subclass recommender.
converter = cattrs.Converter(
    unstruct_collection_overrides={set: list}, forbid_extra_keys=True, use_alias=True
)
"""The default converter for (de-)serializing BayBE-related objects."""

configure_union_passthrough(bool | int | float | str, converter)


def add_type(hook: UnstructureHook) -> UnstructureHook:
    """Wrap a given hook to add type information to the unstructured object."""

    def wrapper(obj: Any, /) -> dict[str, Any]:
        """Unstructure an object and add its type information."""
        result = hook(obj)
        result[_TYPE_FIELD] = obj.__class__.__name__
        return result

    return wrapper


def unstructure_with_type(x: Any, /) -> dict[str, Any]:
    """Unstructure an object and add its type information."""
    return add_type(converter.get_unstructure_hook(x.__class__))(x)


def make_base_structure_hook(base: type[_T]):
    """Create a hook for structuring subclasses using annotations of their base class.

    Reads the ``type`` information from the given input to retrieve the correct
    subclass and then calls the existing structure hook of the that class.
    """

    def structure_base(val: dict | str, cls: type[_T]) -> _T:
        # If the given class can be instantiated, only ensure there is no conflict with
        # a potentially specified type field
        if not is_abstract(cls):
            if (type_ := val.pop(_TYPE_FIELD, None)) and not refers_to(cls, type_):
                raise ValueError(
                    f"The class '{cls.__name__}' specified for deserialization "
                    f"does not match with the given type information '{type_}'."
                )
            concrete_cls = cls

        # Otherwise, extract the type information from the given input and find
        # the corresponding class in the hierarchy
        else:
            type_ = val if isinstance(val, str) else val.pop(_TYPE_FIELD)
            concrete_cls = find_subclass(base, type_)

        fn = converter.get_structure_hook(concrete_cls)
        return fn({} if isinstance(val, str) else val, concrete_cls)

    return structure_base


converter.register_unstructure_hook_func(
    lambda cls: is_abstract(cls) and cls.__module__.startswith("baybe."),
    unstructure_with_type,
)

converter.register_structure_hook_factory(
    lambda cls: is_abstract(cls) and cls.__module__.startswith("baybe."),
    make_base_structure_hook,
)


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


def block_serialization_hook(obj: Any) -> None:  # noqa: DOC101, DOC103
    """Prevent serialization of the passed object.

    Raises:
        NotImplementedError: Always.
    """
    raise NotImplementedError(
        f"Serializing objects of type '{obj.__class__.__name__}' is not supported."
    )


def block_deserialization_hook(_: Any, cls: type) -> None:  # noqa: DOC101, DOC103
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
converter.register_unstructure_hook(pd.DataFrame, _unstructure_dataframe_hook)
converter.register_structure_hook(pd.DataFrame, _structure_dataframe_hook)
converter.register_unstructure_hook(datetime, lambda x: x.isoformat())
converter.register_structure_hook(datetime, lambda x, _: datetime.fromisoformat(x))
converter.register_unstructure_hook(timedelta, lambda x: f"{x.total_seconds()}s")
converter.register_structure_hook(
    timedelta, lambda x, _: timedelta(seconds=float(x.removesuffix("s")))
)
