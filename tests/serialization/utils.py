"""Utilities for serialization tests."""

import inspect
from typing import TypeVar

from typing_extensions import is_protocol

from baybe.serialization.mixin import SerialMixin

Serializable = TypeVar("Serializable", bound=SerialMixin)


def _get_highest_baseclass(cls: type[SerialMixin]) -> type[SerialMixin]:
    """Get the highest base class representing BayBE components."""
    mro = inspect.getmro(cls)
    filtered = [
        c
        for c in mro
        if c.__module__.startswith("baybe.") and c != SerialMixin and not is_protocol(c)
    ]
    return filtered[-1]


def roundtrip(obj: Serializable) -> Serializable:
    """Perform a roundtrip serialization of the given object."""
    cls = _get_highest_baseclass(type(obj))
    string = obj.to_json()
    return cls.from_json(string)


def assert_roundtrip_consistency(obj: SerialMixin) -> None:
    """A serialization roundtrip yields an equivalent object."""
    obj_roundtrip = roundtrip(obj)
    assert obj == obj_roundtrip, (obj, obj_roundtrip)
