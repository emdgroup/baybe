"""Deprecation for constraints."""

from __future__ import annotations

import warnings
from typing import Any


def ContinuousLinearEqualityConstraint(
    parameters: list[str],
    coefficients: list[float] | None = None,
    rhs: float | None = None,
):
    """Return the appropriate new constraint class."""
    warnings.warn(
        "The use of `ContinuousLinearEqualityConstraint` is deprecated and will be"
        "disabled in a future version. Use `ContinuousLinearConstraint` with operator"
        "'=' instead.",
        DeprecationWarning,
    )

    from baybe.constraints.continuous import ContinuousLinearConstraint

    kwargs: dict[Any, Any] = {"parameters": parameters, "operator": "="}
    if coefficients is not None:
        kwargs["coefficients"] = coefficients
    if rhs is not None:
        kwargs["rhs"] = rhs

    return ContinuousLinearConstraint(**kwargs)


def ContinuousLinearInequalityConstraint(
    parameters: list[str],
    coefficients: list[float] | None = None,
    rhs: float | None = None,
):
    """Return the appropriate new constraint class."""
    warnings.warn(
        "The use of `ContinuousLinearInequalityConstraint` is deprecated and will be"
        "disabled in a future version. Use `ContinuousLinearConstraint` with operator"
        "'>=' or '<=' instead.",
        DeprecationWarning,
    )

    from baybe.constraints.continuous import ContinuousLinearConstraint

    kwargs: dict[Any, Any] = {"parameters": parameters, "operator": ">="}
    if coefficients is not None:
        kwargs["coefficients"] = coefficients
    if rhs is not None:
        kwargs["rhs"] = rhs

    return ContinuousLinearConstraint(**kwargs)
