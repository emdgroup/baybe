"""Numerical targets."""

import warnings
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, cast

import numpy as np
import pandas as pd
from attrs import define, field
from numpy.typing import ArrayLike

from baybe.targets.base import Target
from baybe.targets.enum import TargetMode, TargetTransformMode
from baybe.targets.transforms import (
    bell_transform,
    linear_transform,
    triangular_transform,
)
from baybe.utils import Interval, SerialMixin, convert_bounds

_VALID_TRANSFORM_MODES: Dict[TargetMode, Sequence[TargetTransformMode]] = {
    TargetMode.MAX: (TargetTransformMode.LINEAR,),
    TargetMode.MIN: (TargetTransformMode.LINEAR,),
    TargetMode.MATCH: (TargetTransformMode.TRIANGULAR, TargetTransformMode.BELL),
}
"""A mapping from target modes to allowed target transform modes.
If multiple transform modes are allowed, the first entry is used as default option."""


def _get_target_transform(
    mode: TargetMode, transform_mode: TargetTransformMode
) -> Callable[[ArrayLike, float, float], np.ndarray]:
    """Provide the correct target transform for the given modes."""
    if transform_mode is TargetTransformMode.TRIANGULAR:
        return triangular_transform
    if transform_mode is TargetTransformMode.BELL:
        return bell_transform
    if transform_mode is TargetTransformMode.LINEAR:
        if mode is TargetMode.MAX:
            return partial(linear_transform, descending=False)
        elif mode is TargetMode.MIN:
            return partial(linear_transform, descending=True)
    raise RuntimeError("This line should be impossible to reach.")


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

    transform_mode: Optional[TargetTransformMode] = field(
        converter=lambda x: None if x is None else TargetTransformMode(x)
    )
    """An optional target transform mode."""

    @transform_mode.default
    def _default_transform_mode(self) -> Optional[TargetTransformMode]:
        """Provide the default transform mode for bounded targets."""
        if self.bounds.is_closed:
            fun = _VALID_TRANSFORM_MODES[self.mode][0]
            warnings.warn(
                f"The transformation mode for target '{self.name}' "
                f"in '{self.mode.name}' mode has not been specified. "
                f"Setting the bound transform function to '{fun.name}'.",
                UserWarning,
            )
            return fun
        return None

    @bounds.validator
    def _validate_bounds(self, _: Any, bounds: Interval) -> None:  # noqa: DOC101, DOC103
        """Validate the bounds.

        Raises:
            ValueError: If the bounds are finite on one and infinite on the other end.
            ValueError: If the target is in ``MATCH`` mode but the provided bounds
                are infinite.
        """
        # IMPROVE: We could also include half-way bounds, which however don't work
        # for the desirability approach
        if bounds.is_half_open:
            raise ValueError("Bounds must either be finite or infinite on *both* ends.")
        if self.mode is TargetMode.MATCH and not bounds.is_closed:
            raise ValueError(
                f"Target '{self.name}' is in {TargetMode.MATCH.name} mode,"
                f"which requires finite bounds."
            )

    @transform_mode.validator
    def _validate_transform_mode(  # noqa: DOC101, DOC103
        self, _: Any, value: Optional[TargetTransformMode]
    ) -> None:
        """Validate that the given transform is compatible with the specified mode.

        Raises:
            ValueError: If the specified bound transform function and the target mode
                are not compatible.
        """
        if (value is not None) and (value not in _VALID_TRANSFORM_MODES[self.mode]):
            raise ValueError(
                f"You specified bounds for target '{self.name}', but your "
                f"specified bound transform function '{value}' is not compatible "
                f"with the target mode {self.mode}'. It must be one "
                f"of {_VALID_TRANSFORM_MODES[self.mode]}."
            )

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:  # noqa: D102
        # See base class.

        # When (closed) bounds are given, apply the respective transform
        if self.bounds.is_closed:
            func = _get_target_transform(
                # TODO[typing]: For bounded targets (see if clause), the attrs default
                #   ensures there is always a transform mode specified.
                #   Use function overloads to make this explicit.
                self.mode,
                cast(TargetTransformMode, self.transform_mode),
            )
            transformed = pd.DataFrame(
                func(data, *self.bounds.to_tuple()), index=data.index
            )

        # If no bounds are given, simply negate all target values for ``MIN`` mode.
        # For ``MAX`` mode, nothing needs to be done.
        # For ``MATCH`` mode, the validators avoid a situation without specified bounds.
        elif self.mode is TargetMode.MIN:
            transformed = -data

        else:
            transformed = data.copy()

        return transformed
