"""Converter and hooks."""

import base64
import pickle
from collections.abc import Callable
from typing import Any, TypeVar, get_type_hints

import attrs
import cattrs
import pandas as pd
from cattrs.gen import make_dict_structure_fn, make_dict_unstructure_fn
from cattrs.strategies import configure_union_passthrough

from baybe.utils.basic import find_subclass, refers_to
from baybe.utils.boolean import is_abstract

_T = TypeVar("_T")

# TODO: This urgently needs the `forbid_extra_keys=True` flag, which requires us to
#   switch to the cattrs built-in subclass recommender.
# Using GenConverter for built-in overrides for sets, see
# https://catt.rs/en/latest/indepth.html#customizing-collection-unstructuring
converter = cattrs.GenConverter(unstruct_collection_overrides={set: list})
"""The default converter for (de-)serializing BayBE-related objects."""

configure_union_passthrough(bool | int | float | str, converter)


def unstructure_base(base: Any, overrides: dict | None = None) -> dict:
    """Unstructure an object into a dictionary and adds an entry for the class name.

    Args:
        base: The object that should be unstructured.
        overrides: An optional dictionary of cattrs-overrides for certain attributes.

    Returns:
        The unstructured dict with the additional entry.
    """
    # TODO: use include_subclasses (https://github.com/python-attrs/cattrs/issues/434)

    fun = make_dict_unstructure_fn(base.__class__, converter, **(overrides or {}))
    attrs_dict = fun(base)
    return {
        "type": base.__class__.__name__,
        **attrs_dict,
    }


def get_base_structure_hook(
    base: type[_T],
    overrides: dict | None = None,
) -> Callable[[dict | str, type[_T]], _T]:
    """Return a hook for structuring a specified subclass.

    Provides the inverse operation to ``unstructure_base``.

    Args:
        base: The corresponding class.
        overrides: An optional dictionary of cattrs-overrides for certain attributes.

    Returns:
        The hook.
    """
    # TODO: use include_subclasses (https://github.com/python-attrs/cattrs/issues/434)

    def structure_base(val: dict | str, cls: type[_T]) -> _T:
        # If the given class can be instantiated, only ensure there is no conflict with
        # a potentially specified type field
        if not is_abstract(cls):
            if (type_ := val.pop("type", None)) and not refers_to(cls, type_):
                raise ValueError(
                    f"The class '{cls.__name__}' specified for deserialization "
                    f"does not match with the given type information '{type_}'."
                )
            concrete_cls = cls

        # Otherwise, extract the type information from the given input and find
        # the corresponding class in the hierarchy
        else:
            type_ = val if isinstance(val, str) else val.pop("type")
            concrete_cls = find_subclass(base, type_)

        # Create the structuring function for the class and call it
        fn = make_dict_structure_fn(
            concrete_cls, converter, **(overrides or {}), _cattrs_forbid_extra_keys=True
        )
        return fn({} if isinstance(val, str) else val, concrete_cls)

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


# Register custom un-/structure hooks
converter.register_unstructure_hook(pd.DataFrame, _unstructure_dataframe_hook)
converter.register_structure_hook(pd.DataFrame, _structure_dataframe_hook)
