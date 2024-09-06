"""Validation utilities."""

import math
from collections.abc import Callable
from typing import Any

from attrs import Attribute


def validate_not_nan(self: Any, attribute: Attribute, value: Any) -> None:
    """Attrs-compatible validator to forbid 'nan' values."""
    if isinstance(value, float) and math.isnan(value):
        raise ValueError(
            f"The value passed to attribute '{attribute.name}' of class "
            f"'{self.__class__.__name__}' cannot be 'nan'."
        )


def _make_restricted_float_validator(
    allow_nan: bool, allow_inf: bool
) -> Callable[[Any, Attribute, Any], None]:
    """Make an attrs-compatible validator for restricted floats.

    Args:
        allow_nan: If False, validated values cannot be 'nan'.
        allow_inf: If False, validated values cannot be 'inf' or '-inf'.

    Raises:
        ValueError: If no float range restriction is in place.

    Returns:
        The validator.
    """
    if allow_nan and allow_inf:
        raise ValueError(
            "The requested validator would not restrict the float range. "
            "Hence, you can use `attrs.validators.instance_of(float)` instead."
        )

    def validator(self: Any, attribute: Attribute, value: Any) -> None:
        if not isinstance(value, float):
            raise ValueError(
                f"Values assigned to attribute '{attribute.name}' of class "
                f"'{self.__class__.__name__}' must be of type 'float'. "
                f"Given: {value} (type: {type(value)})"
            )
        if not allow_inf and math.isinf(value):
            raise ValueError(
                f"Values assigned to attribute '{attribute.name}' of class "
                f"'{self.__class__.__name__}' cannot be 'inf' or '-inf'."
            )
        if not allow_nan and math.isnan(value):
            raise ValueError(
                f"Values assigned to attribute '{attribute.name}' of class "
                f"'{self.__class__.__name__}' cannot be 'nan'."
            )

    return validator


finite_float = _make_restricted_float_validator(allow_nan=False, allow_inf=False)
"""Validator for finite (i.e., non-nan and non-infinite) floats."""

non_nan_float = _make_restricted_float_validator(allow_nan=False, allow_inf=True)
"""Validator for non-nan floats."""

non_inf_float = _make_restricted_float_validator(allow_nan=True, allow_inf=False)
"""Validator for non-infinite floats."""
