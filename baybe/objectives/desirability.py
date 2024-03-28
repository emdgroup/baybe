from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from attr import define, field
from attr.validators import deep_iterable, instance_of, min_len
from typing_extensions import TypeGuard

from baybe.objectives.base import Objective
from baybe.objectives.enum import CombineFunc
from baybe.targets.base import Target
from baybe.targets.numerical import NumericalTarget
from baybe.utils.numerical import geom_mean


def _normalize_weights(weights: list[float]) -> list[float]:
    """Normalize a collection of weights such that they sum to 100.

    Args:
        weights: The un-normalized weights.

    Returns:
        The normalized weights.
    """
    return (100 * np.asarray(weights) / np.sum(weights)).tolist()


def _is_all_numerical_targets(x: list[Target], /) -> TypeGuard[list[NumericalTarget]]:
    """Typeguard helper function."""
    return all(isinstance(y, NumericalTarget) for y in x)


@define(frozen=True)
class DesirabilityObjective(Objective):
    targets: list[Target] = field(validator=min_len(1))

    weights: list[float] = field(converter=_normalize_weights)

    combine_func: CombineFunc = field(
        default=CombineFunc.GEOM_MEAN, converter=CombineFunc
    )

    @weights.default
    def _default_weights(self) -> list[float]:
        # By default, all targets are equally important.
        return [1.0] * len(self.targets)

    @targets.validator
    def _validate_targets(  # noqa: DOC101, DOC103
        self, _: Any, targets: list[Target]
    ) -> None:
        if not _is_all_numerical_targets(targets):
            raise ValueError(
                f"'{self.__class__.__name__}' currently only supports targets "
                f"of type '{NumericalTarget.__name__}'."
            )
        # Raises a ValueError if there are unbounded targets when using objective mode
        # DESIRABILITY.
        if any(not target.bounds.is_bounded for target in targets):
            raise ValueError(
                "In 'DESIRABILITY' mode for multiple targets, each target must "
                "have bounds defined."
            )

    @weights.validator
    def _validate_weights(  # noqa: DOC101, DOC103
        self, _: Any, weights: list[float]
    ) -> None:
        if weights is None:
            return

        # Assert that weights is a list of numbers
        validator = deep_iterable(instance_of(float), instance_of(list))
        validator(self, _, weights)

        if len(weights) != len(self.targets):
            raise ValueError(
                f"Weights list for your objective has {len(weights)} values, but you "
                f"defined {len(self.targets)} targets."
            )

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        # Perform transformations that are required independent of the mode
        transformed = data[[t.name for t in self.targets]].copy()
        for target in self.targets:
            transformed[target.name] = target.transform(data[target.name])

        # In desirability mode, the targets are additionally combined further into one
        if self.combine_func is CombineFunc.GEOM_MEAN:
            func = geom_mean
        elif self.combine_func is CombineFunc.MEAN:
            func = partial(np.average, axis=1)
        else:
            raise ValueError(
                f"The specified averaging function '{self.combine_func.name}' "
                f"is unknown."
            )

        vals = func(transformed.values, weights=self.weights)
        transformed = pd.DataFrame({"Comp_Target": vals}, index=transformed.index)

        return transformed
