"""Validation utilities."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

import pandas as pd
from attrs import Attribute

if TYPE_CHECKING:
    from baybe.parameters.base import Parameter
    from baybe.targets.base import Target

    _T = TypeVar("_T", bound=Parameter | Target)


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


def get_transform_objects(
    objects: Sequence[_T],
    df: pd.DataFrame,
    allow_missing: bool,
    allow_extra: bool,
) -> list[_T]:
    """Extract the objects relevant for transforming a given dataframe.

    The passed object are assumed to have corresponding columns in the given dataframe,
    identified through their name attribute. The function returns the subset of objects
    that have a corresponding column in the dataframe and thus provide the necessary
    information for transforming the dataframe.

    Args:
        objects: A collection of objects to be considered for transformation (provided
            they have a match in the given dataframe).
        df: The dataframe to be searched for corresponding columns.
        allow_missing: Flag controlling if objects are allowed to have no corresponding
            columns in the dataframe.
        allow_extra: Flag controlling if the dataframe is allowed to have columns
            that have corresponding objects.

    Raises:
        ValueError: If the given objects and dataframe are not compatible
            under the specified values for the Boolean flags.

    Returns:
        The (subset of) objects that need to be considered for the transformation.
    """
    names = [p.name for p in objects]

    if (not allow_missing) and (missing := set(names) - set(df)):  # type: ignore[arg-type]
        raise ValueError(
            f"The object(s) named {missing} cannot be matched against "
            f"the provided dataframe. If you want to transform a subset of "
            f"columns, explicitly set `allow_missing=True`."
        )

    if (not allow_extra) and (extra := set(df) - set(names)):
        raise ValueError(
            f"The provided dataframe column(s) {extra} cannot be matched against"
            f"the given objects. If you want to transform a dataframe "
            f"with additional columns, explicitly set `allow_extra=True'."
        )

    return [p for p in objects if p.name in df]
