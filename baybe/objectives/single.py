"""Functionality for single-target objectives."""

import pandas as pd
from attr import define, field
from attr.validators import instance_of

from baybe.objectives.base import Objective
from baybe.targets.base import Target
from baybe.utils.dataframe import pretty_print_df
from baybe.utils.plotting import create_str_representation


@define(frozen=True, slots=False)
class SingleTargetObjective(Objective):
    """An objective focusing on a single target."""

    _target: Target = field(validator=instance_of(Target), alias="target")
    """The single target considered by the objective."""

    def __str__(self) -> str:
        targets_list = [target.summary() for target in self.targets]
        targets_df = pd.DataFrame(targets_list)

        fields = [
            create_str_representation(
                "Type", [self.__class__.__name__], single_line=True
            ),
            create_str_representation("Targets", [pretty_print_df(targets_df)]),
        ]

        return create_str_representation("Objective", fields)

    @property
    def targets(self) -> tuple[Target, ...]:  # noqa: D102
        # See base class.
        return (self._target,)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:  # noqa: D102
        # See base class.
        target_data = data[[self._target.name]].copy()
        return self._target.transform(target_data)
