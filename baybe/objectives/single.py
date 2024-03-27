from __future__ import annotations

from typing import Any

import pandas as pd
from attr import define, field
from attr.validators import min_len

from baybe.serialization import SerialMixin
from baybe.targets.base import Target
from baybe.targets.numerical import NumericalTarget


@define(frozen=True)
class SingleTargetObjective(SerialMixin):
    # TODO: The class currently directly depends on `NumericalTarget`. Once this
    #   direct dependence is replaced with a dependence on `Target`, the type
    #   annotations should be changed.

    targets: list[Target] = field(validator=min_len(1))

    @targets.validator
    def _validate_targets(  # noqa: DOC101, DOC103
        self, _: Any, targets: list[NumericalTarget]
    ) -> None:
        # Raises a ValueError if multiple targets are specified when using objective
        # mode SINGLE.
        if len(targets) != 1:
            raise ValueError(
                "For objective mode 'SINGLE', exactly one target must be specified."
            )

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        # Perform transformations that are required independent of the mode
        transformed = data[[t.name for t in self.targets]].copy()
        for target in self.targets:
            transformed[target.name] = target.transform(data[target.name])

        return transformed
