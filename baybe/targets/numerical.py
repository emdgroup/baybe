"""Numerical targets."""

import warnings
from functools import partial
from typing import Any, Dict, Optional, Sequence

import pandas as pd
from attrs import define, field

from baybe.targets.base import Target
from baybe.targets.enum import TargetMode, TargetTransform
from baybe.utils import (
    Interval,
    SerialMixin,
    bound_bell,
    bound_linear,
    bound_triangular,
    convert_bounds,
)

_VALID_TRANSFORMS: Dict[TargetMode, Sequence[TargetTransform]] = {
    TargetMode.MAX: (TargetTransform.LINEAR,),
    TargetMode.MIN: (TargetTransform.LINEAR,),
    TargetMode.MATCH: (TargetTransform.TRIANGULAR, TargetTransform.BELL),
}
"""A mapping from target modes to allowed target transforms. If multiple transforms
are allowed, the first entry is used as default option."""


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

    target_transform: Optional[TargetTransform] = field(
        converter=lambda x: None if x is None else TargetTransform(x)
    )
    """An optional target transform."""

    @target_transform.default
    def _default_target_transform(self) -> Optional[TargetTransform]:
        """Provide the default transform for bounded targets."""
        if self.bounds.is_bounded:
            fun = _VALID_TRANSFORMS[self.mode][0]
            warnings.warn(
                f"The transformation mode for target '{self.name}' "
                f"in '{self.mode.name}' mode has not been specified. "
                f"Setting the bound transform function to '{fun.name}'.",
                UserWarning,
            )
            return fun
        return None

    @bounds.validator
    def _validate_bounds(self, _: Any, value: Interval) -> None:  # noqa: DOC101, DOC103
        """Validate the bounds.

        Raises:
            ValueError: If the bounds are finite on one and infinite on the other end.
            ValueError: If the target is in ``MATCH`` mode but the provided bounds
                are infinite.
        """
        # IMPROVE: We could also include half-way bounds, which however don't work
        # for the desirability approach
        if not (value.is_finite or not value.is_bounded):
            raise ValueError("Bounds must either be finite or infinite on *both* ends.")
        if self.mode is TargetMode.MATCH and not value.is_finite:
            raise ValueError(
                f"Target '{self.name}' is in {TargetMode.MATCH.name} mode,"
                f"which requires finite bounds."
            )

    @target_transform.validator
    def _validate_target_transform(  # noqa: DOC101, DOC103
        self, _: Any, value: str
    ) -> None:
        """Validate that the given transform is compatible with the specified mode.

        Raises:
            ValueError: If the specified bound transform function and the target mode
                are not compatible.
        """
        if (value is not None) and (value not in _VALID_TRANSFORMS[self.mode]):
            raise ValueError(
                f"You specified bounds for target '{self.name}', but your "
                f"specified bound transform function '{value}' is not compatible "
                f"with the target mode {self.mode}'. It must be one "
                f"of {_VALID_TRANSFORMS[self.mode]}."
            )

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:  # noqa: D102
        # See base class.

        transformed = data.copy()

        # TODO: potentially introduce an abstract base class for the transforms
        #   -> this would remove the necessity to maintain the following dict
        #   -> also, it would create a common signature, avoiding the `partial` calls

        # Specify all bound transforms
        bounds_transform_funcs = {
            TargetTransform.LINEAR: bound_linear,
            TargetTransform.TRIANGULAR: bound_triangular,
            TargetTransform.BELL: bound_bell,
        }

        # When bounds are given, apply the respective transform
        if self.bounds.is_bounded:
            func = bounds_transform_funcs[self.target_transform]
            if self.mode is TargetMode.MAX:
                func = partial(func, descending=False)
            elif self.mode is TargetMode.MIN:
                func = partial(func, descending=True)
            transformed = func(transformed, *self.bounds.to_tuple())

        # If no bounds are given, simply negate all target values for ``MIN`` mode.
        # For ``MAX`` mode, nothing needs to be done.
        # For ``MATCH`` mode, the validators avoid a situation without specified bounds.
        elif self.mode is TargetMode.MIN:
            transformed = -transformed

        return transformed
