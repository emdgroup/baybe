"""Validation functionality for parameters."""

from collections.abc import Callable, Collection
from typing import Any

import numpy as np
from attrs import Attribute, fields_dict
from attrs.validators import gt, instance_of, lt

from baybe.parameters.base import Parameter


def validate_decorrelation(obj: Parameter, attribute: Attribute, value: float) -> None:
    """Validate that the input represents a valid decorrelation setting."""
    instance_of((bool, float))(obj, attribute, value)
    if isinstance(value, float):
        gt(0.0)(obj, attribute, value)
        lt(1.0)(obj, attribute, value)


def validate_unique_values(  # noqa: DOC101, DOC103
    obj: Parameter, attribute: Attribute, values: Collection[Any]
) -> None:
    """Validate that the input contains unique elements.

    Raises:
        ValueError: If the input contains duplicates.
    """
    if len(set(values)) != len(values):
        raise ValueError(
            f"The '{attribute.alias}' attribute of parameter '{obj.name}' must contain "
            f"unique elements. Given: {values}."
        )


def validate_is_finite(  # noqa: DOC101, DOC103
    obj: Parameter, attribute: Attribute, values: Collection[float]
) -> None:
    """Validate that the input contains no infinity/nan.

    Raises:
        ValueError: If the input contains infinity/nan.
    """
    if not all(np.isfinite(np.asarray(values))):
        raise ValueError(
            f"The '{attribute.alias}' attribute of parameter '{obj.name}' must not "
            f"contain infinity/nan elements. Given: {values}."
        )


def validate_contains_exactly_one(
    value: Any, /
) -> Callable[[Parameter, Attribute, Collection[Any]], None]:
    """Make a validator to check an attribute contains a certain value exactly once."""

    def validator(  # noqa: DOC101, DOC103
        obj: Parameter, attribute: Attribute, values: Collection[float]
    ) -> None:
        """Validate that the input contains a reference element exactly once.

        Raises:
            ValueError: If the input does not contain the reference exactly once.
        """
        if (count := list(values).count(value)) != 1:
            raise ValueError(
                f"The '{attribute.alias}' attribute of parameter '{obj.name}' must "
                f"contain the element '{value}' exactly once. "
                f"Found {count} such elements in the given input: {values}."
            )

    return validator


def validate_equal_length(
    reference_name: str, /
) -> Callable[[Parameter, Attribute, Collection[Any]], None]:
    """Make a validator to check an attribute length against a reference attribute."""

    def validator(obj: Parameter, attribute: Attribute, value: Collection[Any]) -> None:  # noqa: DOC101, DOC103
        """Validate that the input has the same length as the reference attribute.

        Raises:
            ValueError: If the lengths of the two attributes mismatch.
        """
        other_attr = fields_dict(type(obj))[reference_name]
        if len(value) != (ref_len := len(getattr(obj, reference_name))):
            raise ValueError(
                f"The lengths of the attributes '{other_attr.alias}' and "
                f"'{attribute.alias}' do not match for parameter '{obj.name}'. "
                f"Length of '{other_attr.alias}': {ref_len}. "
                f"Length of '{attribute.alias}': {len(value)}."
            )

    return validator
