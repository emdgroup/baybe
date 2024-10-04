"""Deprecation for constraints."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from cattrs.gen import make_dict_structure_fn

from baybe.serialization import converter
from baybe.utils.basic import find_subclass, refers_to
from baybe.utils.boolean import is_abstract

if TYPE_CHECKING:
    from baybe.constraints.base import Constraint


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


def structure_constraints(val: dict, cls: type) -> Constraint:
    """A structure hook taking care of deprecations."""  # noqa: D401 (imperative mood)
    from baybe.constraints.base import Constraint

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

        if type_ == "ContinuousLinearEqualityConstraint":
            return ContinuousLinearEqualityConstraint(**val)
        elif type_ == "ContinuousLinearInequalityConstraint":
            return ContinuousLinearInequalityConstraint(**val)

        concrete_cls = find_subclass(Constraint, type_)

    # Create the structuring function for the class and call it
    fn: Callable = make_dict_structure_fn(
        concrete_cls, converter, _cattrs_forbid_extra_keys=True
    )
    return fn({} if isinstance(val, str) else val, concrete_cls)
