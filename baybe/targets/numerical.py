"""Numerical targets."""

import gc
import warnings
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, cast

import numpy as np
import pandas as pd
from attrs import define, field
from numpy.typing import ArrayLike

from baybe.serialization import SerialMixin
from baybe.targets.base import Target
from baybe.targets.enum import TargetMode, TargetTransformation
from baybe.targets.transforms import (
    bell_transform,
    linear_transform,
    triangular_transform,
)
from baybe.utils.interval import Interval, convert_bounds

_VALID_TRANSFORMATIONS: dict[TargetMode, Sequence[TargetTransformation]] = {
    TargetMode.MAX: (TargetTransformation.LINEAR,),
    TargetMode.MIN: (TargetTransformation.LINEAR,),
    TargetMode.MATCH: (TargetTransformation.TRIANGULAR, TargetTransformation.BELL),
}
"""A mapping from target modes to allowed target transformations.
If multiple transformations are allowed, the first entry is used as default option."""


def _get_target_transformation(
    mode: TargetMode, transformation: TargetTransformation
) -> Callable[[ArrayLike, float, float], np.ndarray]:
    """Provide the transform callable for the given target mode and transform type."""
    if transformation is TargetTransformation.TRIANGULAR:
        return triangular_transform
    if transformation is TargetTransformation.BELL:
        return bell_transform
    if transformation is TargetTransformation.LINEAR:
        if mode is TargetMode.MAX:
            return partial(linear_transform, descending=False)
        if mode is TargetMode.MIN:
            return partial(linear_transform, descending=True)
        raise ValueError(f"Unrecognized target mode: '{mode}'.")
    raise ValueError(f"Unrecognized target transformation: '{transformation}'.")


@define(frozen=True)
class NumericalTarget(Target, SerialMixin):
    """Class for numerical targets."""

    # NOTE: The type annotations of `bounds` are correctly overridden by the attrs
    #   converter. Nonetheless, PyCharm's linter might incorrectly raise a type warning
    #   when calling the constructor. This is a known issue:
    #       https://youtrack.jetbrains.com/issue/PY-34243
    #   Quote from attrs docs:
    #       If a converter’s first argument has a type annotation, that type will
    #       appear in the signature for __init__. A converter will override an explicit
    #       type annotation or type argument.

    mode: TargetMode = field(converter=TargetMode)
    """The target mode."""

    bounds: Interval = field(default=None, converter=convert_bounds)
    """Optional target bounds."""

    transformation: TargetTransformation | None = field(
        converter=lambda x: None if x is None else TargetTransformation(x)
    )
    """An optional target transformation."""

    @transformation.default
    def _default_transformation(self) -> TargetTransformation | None:
        """Provide the default transformation for bounded targets."""
        if self.bounds.is_bounded:
            fun = _VALID_TRANSFORMATIONS[self.mode][0]
            warnings.warn(
                f"The transformation for target '{self.name}' "
                f"in '{self.mode.name}' mode has not been specified. "
                f"Setting the transformation to '{fun.name}'.",
                UserWarning,
            )
            return fun
        return None

    @bounds.validator
    def _validate_bounds(self, _: Any, bounds: Interval) -> None:  # noqa: DOC101, DOC103
        """Validate the bounds.

        Raises:
            ValueError: If the target is defined on a half-bounded interval.
            ValueError: If the target is in ``MATCH`` mode but the provided bounds
                are infinite.
        """
        # IMPROVE: We could also include half-way bounds, which however don't work
        #   for the desirability approach
        if bounds.is_half_bounded:
            raise ValueError("Targets on half-bounded intervals are not supported.")
        if bounds.is_degenerate:
            raise ValueError(
                "The interval specified by the target bounds cannot be degenerate."
            )
        if self.mode is TargetMode.MATCH and not bounds.is_bounded:
            raise ValueError(
                f"Target '{self.name}' is in {TargetMode.MATCH.name} mode,"
                f"which requires finite bounds."
            )

    @transformation.validator
    def _validate_transformation(  # noqa: DOC101, DOC103
        self, _: Any, value: TargetTransformation | None
    ) -> None:
        """Validate that the given transformation is compatible with the specified mode.

        Raises:
            ValueError: If the target transformation and mode are not compatible.
        """
        if (value is not None) and (value not in _VALID_TRANSFORMATIONS[self.mode]):
            raise ValueError(
                f"You specified bounds for target '{self.name}', but your "
                f"specified transformation '{value}' is not compatible "
                f"with the target mode {self.mode}'. It must be one "
                f"of {_VALID_TRANSFORMATIONS[self.mode]}."
            )

    @property
    def _is_transform_normalized(self) -> bool:
        """Indicate if the computational transformation maps to the unit interval."""
        return (self.bounds.is_bounded) and (self.transformation is not None)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:  # noqa: D102
        # See base class.

        # TODO: The method (signature) needs to be refactored, potentially when
        #   enabling multi-target settings. The current input type suggests that passing
        #   dataframes is allowed, but the code was designed for single targets and
        #   desirability objectives, where only one column is present.
        assert data.shape[1] == 1

        # When a transformation is specified, apply it
        if self.transformation is not None:
            func = _get_target_transformation(
                # TODO[typing]: For bounded targets (see if clause), the attrs default
                #   ensures there is always a transformation specified.
                #   Use function overloads to make this explicit.
                self.mode,
                cast(TargetTransformation, self.transformation),
            )
            transformed = pd.DataFrame(
                func(data, *self.bounds.to_tuple()), index=data.index
            )
        else:
            transformed = data.copy()

        return transformed

    def summary(self) -> dict:  # noqa: D102
        # See base class.
        target_dict = dict(
            Type=self.__class__.__name__,
            Name=self.name,
            Mode=self.mode.name,
            Lower_Bound=self.bounds.lower,
            Upper_Bound=self.bounds.upper,
            Transformation=self.transformation.name if self.transformation else "None",
        )
        return target_dict


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
