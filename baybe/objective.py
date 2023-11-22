"""Functionality for defining optimization objectives."""

from __future__ import annotations

from functools import partial
from typing import Any, List, Literal

import numpy as np
import pandas as pd
from attr import define, field
from attr.validators import deep_iterable, in_, instance_of, min_len

from baybe.targets.numerical import NumericalTarget
from baybe.utils import SerialMixin, geom_mean


def _normalize_weights(weights: List[float]) -> List[float]:
    """Normalize a collection of weights such that they sum to 100.

    Args:
        weights: The un-normalized weights.

    Returns:
        The normalized weights.
    """
    return (100 * np.asarray(weights) / np.sum(weights)).tolist()


@define(frozen=True)
class Objective(SerialMixin):
    """Class for managing optimization objectives."""

    # TODO: The class currently directly depends on `NumericalTarget`. Once this
    #   direct dependence is replaced with a dependence on `Target`, the type
    #   annotations should be changed.

    mode: Literal["SINGLE", "DESIRABILITY"] = field()
    """The optimization mode."""

    targets: List[NumericalTarget] = field(validator=min_len(1))
    """The list of targets used for the objective."""

    weights: List[float] = field(converter=_normalize_weights)
    """The weights used to balance the different targets. By default, all
    weights are equally important."""

    combine_func: Literal["MEAN", "GEOM_MEAN"] = field(
        default="GEOM_MEAN", validator=in_(["MEAN", "GEOM_MEAN"])
    )
    """The function used to combine the different targets."""

    @weights.default
    def _default_weights(self) -> List[float]:
        """Create the default weights."""
        # By default, all targets are equally important.
        return [1.0] * len(self.targets)

    @targets.validator
    def _validate_targets(  # noqa: DOC101, DOC103
        self, _: Any, targets: List[NumericalTarget]
    ) -> None:
        """Validate targets depending on the objective mode.

        Raises:
            ValueError: If multiple targets are specified when using objective mode
                ``SINGLE``.
        """
        # Raises a ValueError if multiple targets are specified when using objective
        # mode SINGLE.
        if (self.mode == "SINGLE") and (len(targets) != 1):
            raise ValueError(
                "For objective mode 'SINGLE', exactly one target must be specified."
            )
        # Raises a ValueError if there are unbounded targets when using objective mode
        # DESIRABILITY.
        if self.mode == "DESIRABILITY":
            if any(not target.bounds.is_bounded for target in targets):
                raise ValueError(
                    "In 'DESIRABILITY' mode for multiple targets, each target must "
                    "have bounds defined."
                )

    @weights.validator
    def _validate_weights(  # noqa: DOC101, DOC103
        self, _: Any, weights: List[float]
    ) -> None:
        """Validate target weights.

        Raises:
            ValueError: If the number of weights and the number of targets differ.
        """
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
        """Transform targets from experimental to computational representation.

        Args:
            data: The data to be transformed. Must contain all target values, can
                contain more columns.

        Returns:
            A dataframe with the targets in computational representation. Columns will
            be as in the input (except when objective mode is ``DESIRABILITY``).

        Raises:
            ValueError: If the specified averaging function is unknown.
        """
        # Perform transformations that are required independent of the mode
        transformed = data[[t.name for t in self.targets]].copy()
        for target in self.targets:
            transformed[target.name] = target.transform(data[target.name])

        # In desirability mode, the targets are additionally combined further into one
        if self.mode == "DESIRABILITY":
            if self.combine_func == "GEOM_MEAN":
                func = geom_mean
            elif self.combine_func == "MEAN":
                func = partial(np.average, axis=1)
            else:
                raise ValueError(
                    f"The specified averaging function {self.combine_func} is unknown."
                )

            vals = func(transformed.values, weights=self.weights)
            transformed = pd.DataFrame({"Comp_Target": vals}, index=transformed.index)

        return transformed
