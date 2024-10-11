"""Functionality for single-target objectives."""

import gc

import pandas as pd
from attrs import define, field
from attrs.validators import instance_of
from typing_extensions import override

from baybe.objectives.base import Objective
from baybe.targets.base import Target
from baybe.utils.dataframe import pretty_print_df
from baybe.utils.plotting import to_string
from baybe.utils.validation import get_transform_objects


@define(frozen=True, slots=False)
class SingleTargetObjective(Objective):
    """An objective focusing on a single target."""

    _target: Target = field(validator=instance_of(Target), alias="target")
    """The single target considered by the objective."""

    @override
    def __str__(self) -> str:
        targets_list = [target.summary() for target in self.targets]
        targets_df = pd.DataFrame(targets_list)

        fields = [
            to_string("Type", self.__class__.__name__, single_line=True),
            to_string("Targets", pretty_print_df(targets_df)),
        ]

        return to_string("Objective", *fields)

    @override
    @property
    def targets(self) -> tuple[Target, ...]:
        return (self._target,)

    @override
    def transform(
        self,
        df: pd.DataFrame,
        /,
        *,
        allow_missing: bool = False,
        allow_extra: bool = False,
    ) -> pd.DataFrame:
        # Even for a single target, it is convenient to use the existing machinery
        # instead of re-implementing the validation logic
        targets = get_transform_objects(
            [self._target], df, allow_missing=allow_missing, allow_extra=allow_extra
        )
        target_data = df[[t.name for t in targets]].copy()

        return self._target.transform(target_data)


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
