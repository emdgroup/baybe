"""Numerical targets."""

import gc
import warnings
from collections.abc import Callable, Sequence
from enum import Enum
from functools import partial
from typing import Any, cast

import numpy as np
import pandas as pd
from attrs import define, field
from numpy.typing import ArrayLike
from typing_extensions import override

from baybe.serialization import SerialMixin
from baybe.targets.base import Target
from baybe.utils.interval import Interval


class TargetMode(Enum):
    """Available modes for targets."""

    MIN = "MIN"
    """The target is to be minimized."""

    MAX = "MAX"
    """The target is to be maximized."""

    MATCH = "MATCH"
    """The target should be close to a given value."""


class TargetTransformation(Enum):
    """Available target transformations."""

    LINEAR = "LINEAR"
    """Linear transformation."""

    TRIANGULAR = "TRIANGULAR"
    """Transformation using triangular-shaped function."""

    BELL = "BELL"
    """Transformation using bell-shaped function."""


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
class LegacyTarget(Target, SerialMixin):
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

    # TODO[typing]: https://github.com/python-attrs/attrs/issues/1435
    bounds: Interval = field(default=None, converter=Interval.create)  # type: ignore[misc]
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
        """Validate compatability between transformation, bounds and the mode.

        Raises:
            ValueError: If a target transformation was provided for an unbounded
                target.
            ValueError: If the target transformation and mode are not compatible.
        """
        if (value is not None) and (not self.bounds.is_bounded):
            raise ValueError(
                f"You specified a transformation for target '{self.name}', but "
                f"did not specify any bounds."
            )
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

    @override
    def transform(
        self, series: pd.Series | None = None, /, *, data: pd.DataFrame | None = None
    ) -> pd.Series:
        # >>>>>>>>>> Deprecation
        if not ((series is None) ^ (data is None)):
            raise ValueError(
                "Provide the data to be transformed as first positional argument."
            )

        if data is not None:
            assert data.shape[1] == 1
            series = data.iloc[:, 0]
            warnings.warn(
                "Providing a dataframe via the `data` argument is deprecated and "
                "will be removed in a future version. Please pass your data "
                "in form of a series as positional argument instead.",
                DeprecationWarning,
            )

        # Mypy does not infer from the above that `series` must be a series here
        assert isinstance(series, pd.Series)
        # <<<<<<<<<< Deprecation

        # When a transformation is specified, apply it
        if self.transformation is not None:
            func = _get_target_transformation(
                # TODO[typing]: For bounded targets (see if clause), the attrs default
                #   ensures there is always a transformation specified.
                #   Use function overloads to make this explicit.
                self.mode,
                cast(TargetTransformation, self.transformation),
            )
            transformed = pd.Series(
                func(series, *self.bounds.to_tuple()),
                index=series.index,
                name=series.name,
            )
        else:
            transformed = series.copy()

        return transformed

    @override
    def summary(self) -> dict:
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


def linear_transform(
    arr: ArrayLike, lower: float, upper: float, descending: bool
) -> np.ndarray:
    """Linearly map values in a specified interval ``[lower, upper]`` to ``[0, 1]``.

    Outside the specified interval, the function remains constant.
    That is, 0 or 1, depending on the side and selected mode.

    Args:
        arr: The values to be mapped.
        lower: The lower boundary of the linear mapping interval.
        upper: The upper boundary of the linear mapping interval.
        descending: If ``True``, the function values decrease from 1 to 0 in the
            specified interval. If ``False``, they increase from 0 to 1.

    Returns:
        A new array containing the transformed values.
    """
    arr = np.asarray(arr)
    if descending:
        res = (upper - arr) / (upper - lower)
        res[arr > upper] = 0.0
        res[arr < lower] = 1.0
    else:
        res = (arr - lower) / (upper - lower)
        res[arr > upper] = 1.0
        res[arr < lower] = 0.0

    return res


def triangular_transform(arr: ArrayLike, lower: float, upper: float) -> np.ndarray:
    """Map values to the interval ``[0, 1]`` in a "triangular" fashion.

    The shape of the function is "triangular" in that is 0 outside a specified interval
    and linearly increases to 1 from both interval ends, reaching the value 1 at the
    center of the interval.

    Args:
        arr: The values to be mapped.
        lower: The lower end of the triangle interval. Below, the mapped values are 0.
        upper:The upper end of the triangle interval. Above, the mapped values are 0.

    Returns:
        A new array containing the transformed values.
    """
    arr = np.asarray(arr)
    mid = lower + (upper - lower) / 2
    res = (arr - lower) / (mid - lower)
    res[arr > mid] = (upper - arr[arr > mid]) / (upper - mid)
    res[arr > upper] = 0.0
    res[arr < lower] = 0.0

    return res


def bell_transform(arr: ArrayLike, lower: float, upper: float) -> np.ndarray:
    """Map values to the interval ``[0, 1]`` in a "Gaussian bell" fashion.

    The shape of the function is "Gaussian bell curve", specified through the boundary
    values of the sigma interval. Reaches the maximum value of 1 at the interval center.

    Args:
        arr: The values to be mapped.
        lower: The input value corresponding to the upper sigma interval boundary.
        upper: The input value corresponding to the lower sigma interval boundary.

    Returns:
        A new array containing the transformed values.
    """
    arr = np.asarray(arr)
    mean = np.mean([lower, upper])
    std = (upper - lower) / 2
    res = np.exp(-((arr - mean) ** 2) / (2.0 * std**2))

    return res
