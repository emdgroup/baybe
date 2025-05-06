"""Numerical targets."""

from __future__ import annotations

import gc
import warnings
from collections.abc import Iterable
from typing import Any, cast

import pandas as pd
from attrs import define, evolve, field
from attrs.converters import optional
from attrs.validators import instance_of
from typing_extensions import override

from baybe.exceptions import IncompatibilityError
from baybe.serialization import SerialMixin, converter
from baybe.targets._deprecated import (
    _VALID_TRANSFORMATIONS,
    TargetMode,
    TargetTransformation,
)
from baybe.targets.base import Target
from baybe.targets.transformation import (
    AbsoluteTransformation,
    AffineTransformation,
    BellTransformation,
    ChainedTransformation,
    ClampingTransformation,
    ExponentialTransformation,
    IdentityTransformation,
    LogarithmicTransformation,
    PowerTransformation,
    Transformation,
    convert_transformation,
)
from baybe.utils.interval import Interval, convert_bounds


@define
class _LegacyAPI:
    """Class for parsing legacy targets arguments (for deprecation)."""

    name: str = field(validator=instance_of(str))

    mode: TargetMode = field(converter=TargetMode)

    bounds: Interval = field(default=None, converter=convert_bounds)

    transformation: TargetTransformation | None = field(
        converter=lambda x: None if x is None else TargetTransformation(x)
    )

    @transformation.default
    def _default_transformation(self) -> TargetTransformation | None:
        """Provide the default transformation for bounded targets."""
        if self.bounds.is_bounded:
            return _VALID_TRANSFORMATIONS[self.mode][0]
        return None


def _translate_legacy_arguments(
    mode: TargetMode, bounds: Interval, transformation: TargetTransformation | None
) -> tuple[Transformation | None, bool]:
    """Translate legacy target arguments to modern arguments."""
    if mode in (TargetMode.MAX, TargetMode.MIN):
        if not bounds.is_bounded:
            return (None, mode == TargetMode.MIN)
        else:
            # Use transformation from what would have been the appropriate call
            return (
                NumericalTarget.ramp(
                    "dummy", cutoffs=bounds, descending=mode == TargetMode.MIN
                ).transformation,
                False,
            )
    else:
        modern_transformation: Transformation
        if transformation is TargetTransformation.BELL:
            center = (bounds.upper + bounds.lower) / 2
            width = (bounds.upper - bounds.lower) / 2
            modern_transformation = BellTransformation(center, width)
        else:
            # Use transformation from what would have been the appropriate call
            modern_transformation = cast(
                Transformation,
                NumericalTarget.match_triangular("dummy", bounds).transformation,
            )
        return (modern_transformation, False)


@define(frozen=True, init=False)
class NumericalTarget(Target, SerialMixin):
    """Class for numerical targets."""

    transformation: Transformation | None = field(
        default=None, converter=optional(convert_transformation)
    )
    """An optional target transformation."""

    minimize: bool = field(default=False, validator=instance_of(bool), kw_only=True)
    """Boolean flag indicating if the target is to be minimized."""

    def __init__(  # noqa: DOC301
        self, name: str, *args, **kwargs
    ):
        """Translate legacy target specifications."""
        # Check if legacy or modern API is used
        try:
            self.__attrs_init__(name, *args, **kwargs)
            return
        except Exception:
            pass

        # Now we know that the legacy API is used
        legacy = _LegacyAPI(name, *args, **kwargs)

        warnings.warn(
            "Creating numerical targets by specifying MAX/MIN/MATCH modes has been "
            "deprecated. For now, you do not need to change your code as we "
            "automatically converted your target to the new format. "
            "However, this functionality will be removed in a future version, so "
            "please familiarize yourself with the new API.",
            DeprecationWarning,
        )

        # Translate to modern API
        transformation, minimize = _translate_legacy_arguments(
            legacy.mode, legacy.bounds, legacy.transformation
        )
        self.__attrs_init__(legacy.name, transformation, minimize=minimize)

    @classmethod
    def from_modern_api(
        cls,
        name: str,
        transformation: Transformation | None = None,
        *,
        minimize: bool = False,
    ) -> NumericalTarget:
        """A deprecation helper for creating a target explicitly using the modern API.

        Args:
            name: The name of the target.
            transformation: An optional transformation.
            minimize: Boolean flag indicating if the target should be minimized.

        Returns:
            The created target object.
        """  # noqa: D401
        warnings.warn(
            f"The helper constructor '{cls.from_modern_api.__name__}' is only "
            f"available during the deprecation phase of the legacy target API and can "
            f"to obtain type hints and for IDE autocompletion. Once the "
            f"deprecation phase is over, the regular constructor will take over its "
            f"role with all typing features.",
            DeprecationWarning,
        )

        return cls(name, transformation, minimize=minimize)

    @classmethod
    def from_legacy_api(
        cls,
        name: str,
        mode: TargetMode,
        bounds: None | Iterable | Interval = None,
        transformation: TargetTransformation | None = None,
    ) -> NumericalTarget:
        """A deprecation helper for creating a target explicitly using the legacy API.

        Args:
            name: The name of the target.
            mode: The target mode (MAX, MIN, MATCH).
            bounds: Optional target bounds.
            transformation: An optional target transformation.

        Returns:
            The created target object.
        """  # noqa: D401
        warnings.warn(
            f"The helper constructor '{cls.from_legacy_api.__name__}' is only "
            f"available during the deprecation phase of the legacy target API and can "
            f"to obtain type hints and for IDE autocompletion. Once the "
            f"deprecation phase is over, please switch to the modern API call using "
            f"the regular constructor.",
            DeprecationWarning,
        )

        bounds = convert_bounds(bounds)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            return cls(name, mode, bounds, transformation)

    @classmethod
    def match_triangular(
        cls, name: str, cutoffs: Interval | Iterable[float]
    ) -> NumericalTarget:
        """Create a target to match a given setpoint using a triangular transformation.

        Args:
            name: The name of the target.
            cutoffs: The cutoff values where the output of the triangular transformation
                reaches zero.

        Returns:
            The target with applied triangular matching transformation.
        """
        interval = Interval.create(cutoffs)
        return NumericalTarget(
            name,
            AffineTransformation.from_unit_interval(interval.center, interval.upper)
            + AbsoluteTransformation()
            + AffineTransformation(factor=-1, shift=1)
            + ClampingTransformation(min=0, max=1),
        )

    @classmethod
    def match_bell(cls, name: str, center: float, width: float) -> NumericalTarget:
        """Create a target to match a given setpoint using a bell transformation.

        Args:
            name: The name of the target.
            center: The center point of the bell curve.
            width: The width of the bell curve.

        Returns:
            The target with applied bell matching transformation.
        """
        return NumericalTarget(name, BellTransformation(center, width))

    @classmethod
    def ramp(
        cls, name: str, cutoffs: Interval | Iterable[float], *, descending: bool = False
    ) -> NumericalTarget:
        """Create a target that is affine in a given range and clamped to 0/1 outside.

        Args:
            name: The name of the target.
            cutoffs: The cutoff values defining the affine region.
            descending: Boolean flag indicating if the transformation is ascending
                or descending in the affine region.

        Returns:
            The target with applied clamped affine transformation.
        """
        bounds = Interval.create(cutoffs).to_tuple()
        if descending:
            bounds = bounds[::-1]
        return NumericalTarget(
            name, AffineTransformation.from_unit_interval(*bounds).clamp(0, 1)
        )

    @property
    def is_normalized(self) -> bool:
        """Boolean flag indicating if the target is normalized to the unit interval."""
        return self.get_image() == Interval(0, 1)

    @property
    def total_transformation(self) -> Transformation:
        """The total applied transformation, including potential negation."""
        transformation = self.transformation or IdentityTransformation()
        return transformation.negate() if self.minimize else transformation

    def get_image(self, interval: Interval | None = None, /) -> Interval:
        """Get the image of a certain interval (assuming transformation continuity)."""
        if self.transformation is None:
            return Interval()

        return self.transformation.get_image(interval)

    def _append_transformation(self, transformation: Transformation) -> NumericalTarget:
        """Append a new transformation.

        Args:
            transformation: The transformation to append.

        Returns:
            A new target with the appended transformation.
        """
        return evolve(  # type: ignore[call-arg]
            self,
            transformation=transformation
            if self.transformation is None
            else ChainedTransformation([self.transformation, transformation]),
        )

    def normalize(self) -> NumericalTarget:
        """Normalize the target to the unit interval using an affine transformation.

        Raises:
            IncompatibilityError: If the target is not bounded.

        Returns:
            The normalized target.
        """
        if not self.get_image().is_bounded:
            raise IncompatibilityError("Only bounded targets can be normalized.")

        return self._append_transformation(
            AffineTransformation.from_unit_interval(*self.get_image().to_tuple())
        )

    def clamp(
        self, min: float | None = None, max: float | None = None
    ) -> NumericalTarget:
        """Clamp the target to a given range.

        Args:
            min: The minimum value of the clamping range.
            max: The maximum value of the clamping range.

        Returns:
            The clamped target.
        """
        min = min if min is not None else float("-inf")
        max = max if max is not None else float("inf")
        return self._append_transformation(ClampingTransformation(min, max))

    def log(self) -> NumericalTarget:
        """Apply a logarithmic transformation to the target.

        Returns:
            The target with applied logarithmic transformation.
        """
        return self._append_transformation(LogarithmicTransformation())

    def exp(self) -> NumericalTarget:
        """Apply an exponential transformation to the target.

        Returns:
            The target with applied exponential transformation.
        """
        return self._append_transformation(ExponentialTransformation())

    def power(self, exponent: float) -> NumericalTarget:
        """Apply a power transformation to the target.

        Args:
            exponent: The exponent of the power transformation.

        Returns:
            The target with applied power transformation.
        """
        return self._append_transformation(PowerTransformation(exponent))

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
        if (self.transformation is not None) or self.minimize:
            import torch

            return pd.Series(
                self.total_transformation(torch.from_numpy(series.to_numpy())),
                index=series.index,
                name=series.name,
            )

        return series.copy()

    @override
    def summary(self):
        return {}


# >>> Deprecation >>> #
_hook = converter.get_structure_hook(NumericalTarget)


@converter.register_structure_hook
def _structure_legacy_target_arguments(x: dict[str, Any], _) -> NumericalTarget:
    """Accept legacy target argument for backward compatibility."""
    try:
        return _hook(x, _)
    except Exception:
        return NumericalTarget(**x)  # type: ignore[return-value]


# <<< Deprecation <<< #


gc.collect()
