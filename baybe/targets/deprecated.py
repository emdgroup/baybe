# pylint: disable=invalid-name,import-outside-toplevel

"""Temporary name aliases for backward compatibility."""

from __future__ import annotations

import warnings

from attrs import define

from baybe.objective import Objective as NewObjective


@define
class Objective(NewObjective):
    """An ```Objective``` alias for backward compatibility."""

    def __attrs_pre_init__(self):
        warnings.warn(
            "The use of `baybe.targets.Objective` is deprecated and will be disabled "
            "in a future version. Use `baybe.objective.Objective` instead.",
            DeprecationWarning,
        )
