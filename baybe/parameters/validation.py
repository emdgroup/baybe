"""Validation functionality for parameters."""

from collections.abc import Collection
from typing import Any

from attrs import Attribute
from attrs.validators import gt, instance_of, lt


def validate_decorrelation(obj: Any, attribute: Attribute, value: Any) -> None:
    """Validate the decorrelation."""
    instance_of((bool, float))(obj, attribute, value)
    if isinstance(value, float):
        gt(0.0)(obj, attribute, value)
        lt(1.0)(obj, attribute, value)


def validate_contains_one(  # noqa: DOC101, DOC103
    obj: Any, _: Any, values: Collection[float], use_alias: bool = True
):
    """Validate that ``value`` contains at least one entry equal to ``1.0``.

    Raises:
        ValueError: If ``value`` does not include ``1.0``
    """
    if 1.0 not in values:
        if use_alias:
            raise ValueError(f"{obj.alias} must contain 1.0")
        else:
            raise ValueError(f"{obj.name} must contain 1.0")
