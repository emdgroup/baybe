"""Temporary dispatching functionality for backward compatibility."""

import warnings
from typing import Any, Literal

from baybe.objectives.desirability import DesirabilityObjective
from baybe.objectives.single import SingleTargetObjective
from baybe.targets.base import Target


def Objective(
    mode: Literal["SINGLE", "DESIRABILITY"],
    targets: list[Target],
    weights: list[float] | None = None,
    combine_func: Literal["MEAN", "GEOM_MEAN"] | None = None,
):
    """Return the appropriate new-style class depending on the mode."""
    warnings.warn(
        "The use of `baybe.objective` is deprecated and will be disabled in "
        "a future version. Use the classes defined in `baybe.objectives` instead.",
        DeprecationWarning,
    )

    if mode == "SINGLE":
        if len(targets) != 1:
            raise ValueError("In 'SINGLE' mode, you must provide exactly one target.")
        return SingleTargetObjective(targets[0])

    elif mode == "DESIRABILITY":
        kwargs: dict[str, Any] = {}
        if weights is not None:
            kwargs["weights"] = weights
        if combine_func is not None:
            kwargs["scalarizer"] = combine_func
        return DesirabilityObjective(targets, **kwargs)

    else:
        raise ValueError(f"Unknown mode: {mode}")
