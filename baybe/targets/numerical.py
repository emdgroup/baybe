"""Numerical targets."""

import logging
from functools import partial
from typing import Any, Literal, Optional

import pandas as pd
from attr import define, field

from baybe.targets.base import Target
from baybe.utils import (
    Interval,
    SerialMixin,
    bound_bell,
    bound_linear,
    bound_triangular,
    convert_bounds,
)

_logger = logging.getLogger(__name__)


# TODO: potentially introduce an abstract base class for the transforms
#   -> this would remove the necessity to maintain the following dict
_VALID_TRANSFORMS = {
    "MAX": ["LINEAR"],
    "MIN": ["LINEAR"],
    "MATCH": ["TRIANGULAR", "BELL"],
}


@define(frozen=True)
class NumericalTarget(Target, SerialMixin):
    """Class for numerical targets."""

    # TODO: Introduce mode enum

    # NOTE: The type annotations of `bounds` are correctly overridden by the attrs
    #   converter. Nonetheless, PyCharm's linter might incorrectly raise a type warning
    #   when calling the constructor. This is a known issue:
    #       https://youtrack.jetbrains.com/issue/PY-34243
    #   Quote from attrs docs:
    #       If a converter’s first argument has a type annotation, that type will
    #       appear in the signature for __init__. A converter will override an explicit
    #       type annotation or type argument.

    mode: Literal["MIN", "MAX", "MATCH"] = field()
    """The optimization mode."""

    bounds: Interval = field(default=None, converter=convert_bounds)
    """Bounds of the value of the target."""

    bounds_transform_func: Optional[str] = field()
    """A function for transforming the bounds."""

    @bounds_transform_func.default
    def _default_bounds_transform_func(self) -> Optional[str]:
        """Create the default bounds transform function."""
        if self.bounds.is_bounded:
            fun = _VALID_TRANSFORMS[self.mode][0]
            _logger.warning(
                "The bound transform function for target '%s' in mode '%s' has not "
                "been specified. Setting the bound transform function to '%s'.",
                self.name,
                self.mode,
                fun,
            )
            return fun
        return None

    @bounds.validator
    def _validate_bounds(self, _: Any, value: Interval) -> None:  # noqa: DOC101, DOC103
        """Validate the bounds.

        Raises:
            ValueError: If the bounds are finite on one and infinite on the other end.
            ValueError: If the target is in ``MATCH`` mode but the provided bounds are
                are infinite.
        """
        # IMPROVE: We could also include half-way bounds, which however don't work
        # for the desirability approach
        if not (value.is_finite or not value.is_bounded):
            raise ValueError("Bounds must either be finite or infinite on *both* ends.")
        if self.mode == "MATCH" and not value.is_finite:
            raise ValueError(
                f"Target '{self.name}' is in 'MATCH' mode, which requires "
                f"finite bounds."
            )

    @bounds_transform_func.validator
    def _validate_bounds_transform_func(  # noqa: DOC101, DOC103
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
            "LINEAR": bound_linear,
            "TRIANGULAR": bound_triangular,
            "BELL": bound_bell,
        }

        # When bounds are given, apply the respective transform
        if self.bounds.is_bounded:
            func = bounds_transform_funcs[self.bounds_transform_func]
            if self.mode == "MAX":
                func = partial(func, descending=False)
            elif self.mode == "MIN":
                func = partial(func, descending=True)
            transformed = func(transformed, *self.bounds.to_tuple())

        # If no bounds are given, simply negate all target values for "MIN" mode.
        # For "MAX" mode, nothing needs to be done.
        # For "MATCH" mode, the validators avoid a situation without specified bounds.
        elif self.mode == "MIN":
            transformed = -transformed

        return transformed
