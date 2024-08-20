"""Functionality for single-target objectives."""

import pandas as pd
from attr import define, field
from attr.validators import instance_of

from baybe.objectives.base import Objective
from baybe.targets.base import Target


@define(frozen=True, slots=False)
class SingleTargetObjective(Objective):
    """An objective focusing on a single target."""

    _target: Target = field(validator=instance_of(Target), alias="target")
    """The single target considered by the objective."""

    def __str__(self) -> str:
        targets_list = [target.summary() for target in self.targets]
        targets_df = pd.DataFrame(targets_list)

        objective_str = f"""Objective
        \nType: {self.__class__.__name__}
        \nTargets \n{targets_df}"""

        return objective_str.replace("\n", "\n ")

    @property
    def targets(self) -> tuple[Target, ...]:  # noqa: D102
        # See base class.
        return (self._target,)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:  # noqa: D102
        # See base class.
        target_data = data[[self._target.name]].copy()
        return self._target.transform(target_data)
