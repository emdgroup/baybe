"""Validation functionality for parameters."""

from typing import Any

from attrs import Attribute
from attrs.validators import gt, instance_of, lt


def validate_decorrelation(obj: Any, attribute: Attribute, value: Any) -> None:
    """Validate the decorrelation."""
    instance_of((bool, float))(obj, attribute, value)
    if isinstance(value, float):
        gt(0.0)(obj, attribute, value)
        lt(1.0)(obj, attribute, value)
