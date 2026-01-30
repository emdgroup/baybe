"""Validation functionality for parameters."""

from collections.abc import Collection

from attrs import Attribute
from attrs.validators import gt, instance_of, lt

from baybe.parameters.base import Parameter


def validate_decorrelation(obj: Parameter, attribute: Attribute, value: float) -> None:
    """Validate that the input represents a valid decorrelation setting."""
    instance_of((bool, float))(obj, attribute, value)
    if isinstance(value, float):
        gt(0.0)(obj, attribute, value)
        lt(1.0)(obj, attribute, value)


def validate_contains_exactly_one_zero(  # noqa: DOC101, DOC103
    obj: Parameter, attribute: Attribute, values: Collection[float]
) -> None:
    """Validate that the input contains exactly one element equal to ``0.0``.

    Raises:
        ValueError: If the input does not contain ``0.0`` exactly once.
    """
    if (count := list(values).count(0.0)) != 1:
        raise ValueError(
            f"The '{attribute.alias}' attribute of parameter '{obj.name}' must contain "
            f"the element 0.0 exactly once. Found {count} such elements: {values}."
        )


def validate_contains_one(  # noqa: DOC101, DOC103
    obj: Parameter, attribute: Attribute, values: Collection[float]
):
    """Validate that ``value`` contains at least one entry equal to ``1.0``.

    Raises:
        ValueError: If ``value`` does not include ``1.0``
    """
    if 1.0 not in values:
        raise ValueError(
            f"The '{attribute.alias}' attribute of parameter '{obj.name}' must "
            f"contain the element 1.0. Given: {values}."
        )
