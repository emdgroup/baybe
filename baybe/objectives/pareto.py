"""Functionality for multi-target objectives."""

from __future__ import annotations

import gc
from typing import ClassVar

from attrs import define, field
from attrs.validators import deep_iterable, instance_of, min_len
from typing_extensions import override

from baybe.objectives.base import Objective
from baybe.objectives.validation import validate_target_names
from baybe.targets.numerical import NumericalTarget
from baybe.utils.basic import to_tuple


@define(frozen=True, slots=False)
class ParetoObjective(Objective):
    """An objective handling multiple targets in a Pareto sense."""

    is_multi_output: ClassVar[bool] = True
    # See base class.

    _targets: tuple[NumericalTarget, ...] = field(
        converter=to_tuple,
        validator=[
            min_len(2),
            deep_iterable(member_validator=instance_of(NumericalTarget)),
            validate_target_names,
        ],
        alias="targets",
    )
    "The targets considered by the objective."

    @override
    @property
    def targets(self) -> tuple[NumericalTarget, ...]:
        return self._targets

    @override
    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(target.name for target in self.targets)

    @override
    @property
    def supports_partial_measurements(self) -> bool:
        return True


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
