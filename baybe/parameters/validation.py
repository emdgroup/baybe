"""Validation functionality for parameters."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from attrs.validators import gt, instance_of, lt


def validate_unique_values(  # noqa: DOC101, DOC103
    obj: Any, _: Any, value: list
) -> None:
    """Validate that there are no duplicates in ``value``.

    Raises:
        ValueError: If there are duplicates in ``value``.
    """
    if len(set(value)) != len(value):
        raise ValueError(
            f"Cannot assign the following values containing duplicates to "
            f"parameter {obj.name}: {value}."
        )


def validate_decorrelation(obj: Any, attribute: Any, value: float) -> None:
    """Validate the decorrelation."""
    instance_of((bool, float))(obj, attribute, value)
    if isinstance(value, float):
        gt(0.0)(obj, attribute, value)
        lt(1.0)(obj, attribute, value)


def validate_is_finite(  # noqa: DOC101, DOC103
    obj: Any, _: Any, value: Sequence[float]
) -> None:
    """Validate that ``value`` contains no infinity/nan.

    Raises:
        ValueError: If ``value`` contains infinity/nan.
    """
    if not all(np.isfinite(value)):
        raise ValueError(
            f"Cannot assign the following values containing infinity/nan to "
            f"parameter {obj.name}: {value}."
        )
