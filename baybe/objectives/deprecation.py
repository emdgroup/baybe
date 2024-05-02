"""Temporary functionality for backward compatibility."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from cattrs import override
from cattrs.gen import make_dict_structure_fn

from baybe.serialization import converter
from baybe.targets.base import Target
from baybe.utils.basic import get_subclasses
from baybe.utils.boolean import is_abstract

if TYPE_CHECKING:
    from baybe.objectives.base import Objective


def structure_objective(val: dict, cls: type) -> Objective:
    """A structure hook that automatically determines an objective fallback type."""  # noqa: D401 (imperative mood)
    from baybe.objective import Objective as OldObjective
    from baybe.objectives.base import Objective
    from baybe.objectives.desirability import DesirabilityObjective
    from baybe.objectives.single import SingleTargetObjective

    val = val.copy()

    # Use the specified objective type
    try:
        if is_abstract(cls):
            _type = val.pop("type")
            try:
                cls = next(
                    cl for cl in get_subclasses(Objective) if cl.__name__ == _type
                )
            except StopIteration as ex:
                raise ValueError(f"Unknown subclass '{_type}'.") from ex
        fun = make_dict_structure_fn(
            cls,
            converter,
            _cattrs_forbid_extra_keys=True,
            _target=override(rename="target"),
            _targets=override(rename="targets"),
        )  # type: ignore
        return fun(val, cls)

    # If no type is provided, determine the type by the number of targets given
    except KeyError:
        n_targets = len(val["targets"])
        if n_targets == 0:
            raise ValueError("No targets are specified.")
        elif n_targets == 1:
            cls = SingleTargetObjective
        else:
            cls = DesirabilityObjective
        warnings.warn(
            f"An objective has been specified without a corresponding type. "
            f"Since {n_targets} target(s) have been provided, "
            f"'{cls.__name__}' is used as a fallback. "
            f"However, this behavior is deprecated and will be disabled in "
            f"a future version.",
            DeprecationWarning,
        )
        val["targets"] = converter.structure(val["targets"], list[Target])
        return OldObjective(**val)
