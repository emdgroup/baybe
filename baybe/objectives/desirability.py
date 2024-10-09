"""Functionality for desirability objectives."""

import gc
from collections.abc import Callable
from functools import cached_property, partial
from typing import TypeGuard

import cattrs
import numpy as np
import numpy.typing as npt
import pandas as pd
from attrs import define, field
from attrs.validators import deep_iterable, gt, instance_of, min_len

from baybe.objectives.base import Objective
from baybe.objectives.enum import Scalarizer
from baybe.targets.base import Target
from baybe.targets.numerical import NumericalTarget
from baybe.utils.basic import to_tuple
from baybe.utils.dataframe import pretty_print_df
from baybe.utils.numerical import geom_mean
from baybe.utils.plotting import to_string
from baybe.utils.validation import finite_float


def _is_all_numerical_targets(
    x: tuple[Target, ...], /
) -> TypeGuard[tuple[NumericalTarget, ...]]:
    """Typeguard helper function."""
    return all(isinstance(y, NumericalTarget) for y in x)


def scalarize(
    values: npt.ArrayLike, scalarizer: Scalarizer, weights: npt.ArrayLike
) -> np.ndarray:
    """Scalarize the rows of a 2-D array, producing a 1-D array.

    Args:
        values: The 2-D array whose rows are to be scalarized.
        scalarizer: The scalarization mechanism to be used.
        weights: Weights for the columns of the input array.

    Raises:
        ValueError: If the provided array is not two-dimensional.
        NotImplementedError: If the requested scalarizer is not implemented.

    Returns:
        np.ndarray: A 1-D array containing the scalarized values.
    """
    if np.ndim(values) != 2:
        raise ValueError("The provided array must be two-dimensional.")

    func: Callable

    if scalarizer is Scalarizer.GEOM_MEAN:
        func = geom_mean
    elif scalarizer is Scalarizer.MEAN:
        func = partial(np.average, axis=1)
    else:
        raise NotImplementedError(
            f"No scalarization mechanism defined for '{scalarizer.name}'."
        )

    return func(values, weights=weights)


@define(frozen=True, slots=False)
class DesirabilityObjective(Objective):
    """An objective scalarizing multiple targets using desirability values."""

    _targets: tuple[Target, ...] = field(
        converter=to_tuple,
        validator=[min_len(2), deep_iterable(member_validator=instance_of(Target))],
        alias="targets",
    )
    "The targets considered by the objective."

    weights: tuple[float, ...] = field(
        converter=lambda w: cattrs.structure(w, tuple[float, ...]),
        validator=deep_iterable(member_validator=[finite_float, gt(0.0)]),
    )
    """The weights to balance the different targets.
    By default, all targets are considered equally important."""

    scalarizer: Scalarizer = field(default=Scalarizer.GEOM_MEAN, converter=Scalarizer)
    """The mechanism to scalarize the weighted desirability values of all targets."""

    @weights.default
    def _default_weights(self) -> tuple[float, ...]:
        """Create unit weights for all targets."""
        return tuple(1.0 for _ in range(len(self.targets)))

    @_targets.validator
    def _validate_targets(self, _, targets) -> None:  # noqa: DOC101, DOC103
        if not _is_all_numerical_targets(targets):
            raise TypeError(
                f"'{self.__class__.__name__}' currently only supports targets "
                f"of type '{NumericalTarget.__name__}'."
            )
        if len({t.name for t in targets}) != len(targets):
            raise ValueError("All target names must be unique.")
        if not all(target._is_transform_normalized for target in targets):
            raise ValueError(
                "All targets must have normalized computational representations to "
                "enable the computation of desirability values. This requires having "
                "appropriate target bounds and transformations in place."
            )

    @weights.validator
    def _validate_weights(self, _, weights) -> None:  # noqa: DOC101, DOC103
        if (lw := len(weights)) != (lt := len(self.targets)):
            raise ValueError(
                f"If custom weights are specified, there must be one for each target. "
                f"Specified number of targets: {lt}. Specified number of weights: {lw}."
            )

    @property
    def targets(self) -> tuple[Target, ...]:  # noqa: D102
        # See base class.
        return self._targets

    @cached_property
    def _normalized_weights(self) -> np.ndarray:
        """The normalized target weights."""
        return np.asarray(self.weights) / np.sum(self.weights)

    def __str__(self) -> str:
        targets_list = [target.summary() for target in self.targets]
        targets_df = pd.DataFrame(targets_list)
        targets_df["Weight"] = self.weights

        fields = [
            to_string("Type", self.__class__.__name__, single_line=True),
            to_string("Targets", pretty_print_df(targets_df)),
            to_string("Scalarizer", self.scalarizer.name, single_line=True),
        ]

        return to_string("Objective", *fields)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:  # noqa: D102
        # See base class.

        # Transform all targets individually
        transformed = data[[t.name for t in self.targets]].copy()
        for target in self.targets:
            transformed[target.name] = target.transform(data[[target.name]])

        # Scalarize the transformed targets into desirability values
        vals = scalarize(transformed.values, self.scalarizer, self._normalized_weights)

        # Store the total desirability in a dataframe column
        transformed = pd.DataFrame({"Desirability": vals}, index=transformed.index)

        return transformed


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
