"""Numerical targets."""

from __future__ import annotations

import gc
import warnings
from collections.abc import Sequence
from operator import add, mul, sub
from typing import Any, cast

import pandas as pd
from attrs import define, evolve, field
from attrs.validators import instance_of
from typing_extensions import override

from baybe.exceptions import IncompatibilityError
from baybe.serialization import SerialMixin, converter
from baybe.serialization.core import select_constructor_hook
from baybe.targets._deprecated import (
    _VALID_TRANSFORMATIONS,
    TargetMode,
    TargetTransformation,
)
from baybe.targets.base import Target
from baybe.targets.enum import MatchMode
from baybe.targets.utils import (
    combine_numerical_targets,
)
from baybe.transformations import (
    AbsoluteTransformation,
    AffineTransformation,
    BellTransformation,
    ChainedTransformation,
    ClampingTransformation,
    ExponentialTransformation,
    IdentityTransformation,
    LogarithmicTransformation,
    PowerTransformation,
    SigmoidTransformation,
    Transformation,
    TriangularTransformation,
    convert_transformation,
)
from baybe.utils.basic import UncertainBool
from baybe.utils.interval import ConvertibleToInterval, Interval
from baybe.utils.metadata import (
    ConvertibleToMeasurableMetadata,
    MeasurableMetadata,
    to_metadata,
)


@define
class _LegacyInterface:
    """Class for parsing legacy targets arguments (for deprecation)."""

    name: str = field(validator=instance_of(str))

    mode: TargetMode = field(converter=TargetMode)

    # TODO[typing]: https://github.com/python-attrs/attrs/issues/1435
    bounds: Interval = field(default=None, converter=Interval.create)  # type: ignore[misc]

    transformation: TargetTransformation | None = field(
        converter=lambda x: None if x is None else TargetTransformation(x),
    )

    metadata: MeasurableMetadata = field(
        factory=MeasurableMetadata,
        converter=lambda x: to_metadata(x, MeasurableMetadata),
        kw_only=True,
    )

    @transformation.default
    def _default_transformation(self) -> TargetTransformation | None:
        """Provide the default transformation for bounded targets."""
        if self.bounds.is_bounded:
            return _VALID_TRANSFORMATIONS[self.mode][0]
        return None


def _translate_legacy_arguments(
    mode: TargetMode, bounds: Interval, transformation: TargetTransformation | None
) -> tuple[Transformation, bool]:
    """Translate legacy target arguments to modern arguments."""
    if mode in (TargetMode.MAX, TargetMode.MIN):
        if not bounds.is_bounded:
            return (IdentityTransformation(), mode == TargetMode.MIN)
        else:
            # Use transformation from what would have been the appropriate call
            return (
                NumericalTarget.normalized_ramp(
                    "dummy", cutoffs=bounds, descending=mode == TargetMode.MIN
                ).transformation,
                False,
            )
    else:
        modern_transformation: Transformation
        if transformation is TargetTransformation.BELL:
            width = (bounds.upper - bounds.lower) / 2
            modern_transformation = BellTransformation(bounds.center, width)
        else:
            # Use transformation from what would have been the appropriate calls
            modern_transformation = cast(
                Transformation,
                NumericalTarget.match_triangular(
                    "dummy", cutoffs=bounds
                ).transformation,
            )
        return (modern_transformation, False)


@define(frozen=True, init=False)
class NumericalTarget(Target, SerialMixin):
    """Class for numerical targets."""

    transformation: Transformation = field(
        factory=IdentityTransformation, converter=convert_transformation
    )
    """An optional target transformation."""

    minimize: bool = field(default=False, validator=instance_of(bool), kw_only=True)
    """Boolean flag indicating if the target is to be minimized."""

    def __init__(  # noqa: DOC301
        self, name: str, *args, _enforce_modern_interface: bool = False, **kwargs
    ):
        """Translate legacy target specifications."""
        # Check if legacy or modern interface is used
        try:
            self.__attrs_init__(name, *args, **kwargs)
            return
        except TypeError as ex:
            if _enforce_modern_interface:
                raise ex

        # Now we know that the legacy interface is used
        legacy = _LegacyInterface(name, *args, **kwargs)

        warnings.warn(
            "Creating numerical targets by specifying MAX/MIN/MATCH modes has been "
            "deprecated. For now, you do not need to change your code as we "
            "automatically converted your target to the new format. "
            "However, this functionality will be removed in a future version, so "
            "please familiarize yourself with the new interface.",
            DeprecationWarning,
        )

        # Translate to modern interface
        transformation, minimize = _translate_legacy_arguments(
            legacy.mode, legacy.bounds, legacy.transformation
        )
        metadata = legacy.metadata
        self.__attrs_init__(
            legacy.name, transformation, minimize=minimize, metadata=metadata
        )

    def __neg__(self) -> NumericalTarget:
        return self.negate()

    def __add__(self, other: Any) -> NumericalTarget:
        if isinstance(other, (int, float)):
            return self._append_transformation(AffineTransformation(shift=other))
        if isinstance(other, NumericalTarget):
            return combine_numerical_targets(self, other, operator=add)
        return NotImplemented

    def __sub__(self, other: Any) -> NumericalTarget:
        if isinstance(other, (int, float)):
            return self._append_transformation(AffineTransformation(shift=-other))
        if isinstance(other, NumericalTarget):
            return combine_numerical_targets(self, other, operator=sub)
        return NotImplemented

    def __mul__(self, other: Any) -> NumericalTarget:
        if isinstance(other, (int, float)):
            return self._append_transformation(AffineTransformation(factor=other))
        if isinstance(other, NumericalTarget):
            return combine_numerical_targets(self, other, operator=mul)
        return NotImplemented

    def __truediv__(self, other: Any) -> NumericalTarget:
        if isinstance(other, (int, float)):
            return self._append_transformation(AffineTransformation(factor=1 / other))
        return NotImplemented

    @classmethod
    def from_modern_interface(
        cls,
        name: str,
        transformation: Transformation | None = None,
        *,
        minimize: bool = False,
        metadata: ConvertibleToMeasurableMetadata = None,
    ) -> NumericalTarget:
        """A deprecation helper for creating targets using the modern interface.

        Args:
            name: The name of the target.
            transformation: An optional transformation.
            minimize: Boolean flag indicating if the target should be minimized.
            metadata: Optional metadata containing description, unit, and other
                information.

        Returns:
            The created target object.
        """  # noqa: D401
        warnings.warn(
            f"The helper constructor '{cls.from_modern_interface.__name__}' is only "
            f"available during the deprecation phase of the legacy target interface "
            f"to provide type hints and for IDE autocompletion. Once the "
            f"deprecation phase is over, the regular constructor will take over its "
            f"role with all typing features.",
            DeprecationWarning,
        )

        return (
            cls(name, minimize=minimize, _enforce_modern_interface=True)
            if transformation is None
            else cls(
                name,
                transformation,
                minimize=minimize,
                metadata=metadata,
                _enforce_modern_interface=True,
            )
        )

    @classmethod
    def from_legacy_interface(
        cls,
        name: str,
        mode: TargetMode,
        bounds: ConvertibleToInterval = None,
        transformation: TargetTransformation | None = None,
        *,
        metadata: ConvertibleToMeasurableMetadata = None,
    ) -> NumericalTarget:
        """A deprecation helper for creating targets using the legacy interface.

        Args:
            name: The name of the target.
            mode: The target mode (MAX, MIN, MATCH).
            bounds: Optional target bounds.
            transformation: An optional target transformation.
            metadata: Optional metadata containing description, unit, and other
                information.

        Returns:
            The created target object.
        """  # noqa: D401
        warnings.warn(
            f"The helper constructor '{cls.from_legacy_interface.__name__}' is only "
            f"available during the deprecation phase of the legacy target interface "
            f"to provide type hints and for IDE autocompletion. Once the "
            f"deprecation phase is over, please switch to the modern interface "
            f"using the regular constructor call.",
            DeprecationWarning,
        )

        bounds = Interval.create(bounds)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            return cls(name, mode, bounds, transformation, metadata=metadata)

    @classmethod
    def match_absolute(
        cls,
        name: str,
        match_value: float,
        *,
        mismatch_instead: bool = False,
        match_mode: MatchMode | str = MatchMode.EQ,
        metadata: ConvertibleToMeasurableMetadata = None,
    ) -> NumericalTarget:
        """Create a target to match a given value using an absolute transformation.

        Args:
            name: The name of the target.
            match_value: The value to be matched.
            mismatch_instead: If ``True``, the target will instead seek to maximize
                the distance to the given ``match_value``.
            match_mode: The matching mode to be used. See
                :class:`baybe.targets.enum.MatchMode`.
            metadata: See :class:`baybe.targets.numerical.NumericalTarget`.

        Returns:
            The target with applied absolute matching transformation.
        """
        return NumericalTarget(
            name,
            AffineTransformation(shift=-match_value) | AbsoluteTransformation(),
            minimize=not mismatch_instead,
            metadata=metadata,
        )._hold_output(match_value, match_mode)

    @classmethod
    def match_quadratic(
        cls,
        name: str,
        match_value: float,
        *,
        mismatch_instead: bool = False,
        match_mode: MatchMode | str = MatchMode.EQ,
        metadata: ConvertibleToMeasurableMetadata = None,
    ) -> NumericalTarget:
        """Create a target to match a given value using a quadratic transformation.

        Args:
            name: The name of the target.
            match_value: The value to be matched.
            mismatch_instead: If ``True``, the target will instead seek to maximize
                the distance to the given ``match_value``.
            match_mode: The matching mode to be used. See
                :class:`baybe.targets.enum.MatchMode`.
            metadata: See :class:`baybe.targets.numerical.NumericalTarget`.

        Returns:
            The target with applied quadratic matching transformation.
        """
        return NumericalTarget.match_power(
            name,
            match_value,
            exponent=2,
            mismatch_instead=mismatch_instead,
            match_mode=match_mode,
            metadata=metadata,
        )

    @classmethod
    def match_power(
        cls,
        name: str,
        match_value: float,
        exponent: int,
        *,
        mismatch_instead: bool = False,
        match_mode: MatchMode | str = MatchMode.EQ,
        metadata: ConvertibleToMeasurableMetadata = None,
    ) -> NumericalTarget:
        """Create a target to match a given value using a power transformation.

        Args:
            name: The name of the target.
            match_value: The value to be matched.
            exponent: The exponent of applied the power transformation.
            mismatch_instead: If ``True``, the target will instead seek to maximize
                the distance to the given ``match_value``.
            match_mode: The matching mode to be used. See
                :class:`baybe.targets.enum.MatchMode`.
            metadata: See :class:`baybe.targets.numerical.NumericalTarget`.

        Returns:
            The target with applied power matching transformation.
        """
        return NumericalTarget(
            name,
            AffineTransformation(shift=-match_value)
            | AbsoluteTransformation()
            | PowerTransformation(exponent),
            minimize=not mismatch_instead,
            metadata=metadata,
        )._hold_output(match_value, match_mode)

    @classmethod
    def match_triangular(
        cls,
        name: str,
        match_value: float | None = None,
        *,
        cutoffs: ConvertibleToInterval = None,
        width: float | None = None,
        margins: Sequence[float] | None = None,
        mismatch_instead: bool = False,
        match_mode: MatchMode | str = MatchMode.EQ,
        metadata: ConvertibleToMeasurableMetadata = None,
    ) -> NumericalTarget:
        """Create a target to match a given value using a triangular transformation.

        Args:
            name: The name of the target.
            match_value: The value to be matched. Can be omitted when ``cutoffs`` are
                provided, in which case it defaults to the midpoint.
            cutoffs: The cutoff values where the output of the transformation
                reaches zero.
            width: The width of the (symmetric) triangular transformation.
            margins: The margins defining how far the triangle extends in both
                directions.
            mismatch_instead: If ``True``, the target will instead seek to maximize
                the distance to the given ``match_value``.
            match_mode: The matching mode to be used. See
                :class:`baybe.targets.enum.MatchMode`.
            metadata: See :class:`baybe.targets.numerical.NumericalTarget`.

        Raises:
            ValueError: If more than one of ``cutoffs``, ``width``, or ``margins`` is
                provided.

        Returns:
            The target with applied triangular matching transformation.
        """
        if match_value is None:
            if cutoffs is None:
                raise ValueError(
                    "If no 'match_value' is provided, 'cutoffs' must be specified."
                )
            cutoffs = Interval.create(cutoffs)
            match_value = cutoffs.center

        if sum(x is not None for x in (cutoffs, width, margins)) != 1:
            raise ValueError(
                "Exactly one of 'cutoffs', 'width', or 'margins' must be provided."
            )

        if cutoffs is not None:
            transformation = TriangularTransformation(cutoffs, match_value)
        elif width is not None:
            transformation = TriangularTransformation.from_width(match_value, width)
        elif margins is not None:
            transformation = TriangularTransformation.from_margins(match_value, margins)

        return NumericalTarget(
            name, transformation, minimize=mismatch_instead, metadata=metadata
        )._hold_output(match_value, match_mode)

    @classmethod
    def match_bell(
        cls,
        name: str,
        match_value: float,
        sigma: float,
        *,
        mismatch_instead: bool = False,
        match_mode: MatchMode | str = MatchMode.EQ,
        metadata: ConvertibleToMeasurableMetadata = None,
    ) -> NumericalTarget:
        """Create a target to match a given value using a bell transformation.

        Args:
            name: The name of the target.
            match_value: The value to be matched.
            sigma: The scale parameter controlling the width of the bell curve. For more
                details, see :class:`baybe.transformations.basic.BellTransformation`.
            mismatch_instead: If ``True``, the target will instead seek to maximize
                the distance to the given ``match_value``.
            match_mode: The matching mode to be used. See
                :class:`baybe.targets.enum.MatchMode`.
            metadata: See :class:`baybe.targets.numerical.NumericalTarget`.

        Returns:
            The target with applied bell matching transformation.
        """
        return NumericalTarget(
            name,
            BellTransformation(match_value, sigma),
            minimize=mismatch_instead,
            metadata=metadata,
        )._hold_output(match_value, match_mode)

    @classmethod
    def normalized_ramp(
        cls,
        name: str,
        cutoffs: ConvertibleToInterval,
        *,
        descending: bool = False,
        metadata: ConvertibleToMeasurableMetadata = None,
    ) -> NumericalTarget:
        """Create a target that is affine in a given range and clamped to 0/1 outside.

        Args:
            name: The name of the target.
            cutoffs: The cutoff values defining the affine region.
            descending: Boolean flag indicating if the transformation is ascending
                or descending in the affine region.
            metadata: See :class:`baybe.targets.numerical.NumericalTarget`.

        Returns:
            The target with applied clamped affine transformation.
        """
        bounds = Interval.create(cutoffs).to_tuple()
        if descending:
            bounds = bounds[::-1]
        return NumericalTarget(
            name,
            AffineTransformation.from_values_mapped_to_unit_interval(*bounds).clamp(
                0, 1
            ),
            metadata=metadata,
        )

    @classmethod
    def normalized_sigmoid(
        cls,
        name: str,
        anchors: Sequence[Sequence[float]],
        *,
        metadata: ConvertibleToMeasurableMetadata = None,
    ) -> NumericalTarget:
        """Create a sigmoid-transformed target.

        Args:
            name: The name of the target.
            anchors: See :class:`baybe.transformations.basic.SigmoidTransformation`.
            metadata: See :class:`baybe.targets.numerical.NumericalTarget`.

        Returns:
            The target with applied sigmoid transformation.
        """
        return NumericalTarget(
            name, SigmoidTransformation.from_anchors(anchors), metadata=metadata
        )

    @property
    def is_normalized(self) -> UncertainBool:
        """Boolean flag indicating if the target is normalized to the unit interval."""
        return UncertainBool.from_erroneous_callable(
            lambda: self.get_image() == Interval(0, 1)
        )

    def get_codomain(self, interval: Interval | None = None, /) -> Interval:
        """Get the codomain of an interval (assuming transformation continuity)."""
        return self.transformation.get_codomain(interval)

    def get_image(self, interval: Interval | None = None, /) -> Interval:
        """Get the image of an interval (assuming transformation continuity)."""
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
            transformation=ChainedTransformation([self.transformation, transformation]),
        )

    def negate(self) -> NumericalTarget:
        """Apply a negation transformation to the target.

        Returns:
            The target with applied negation transformation.
        """
        return self._append_transformation(AffineTransformation(factor=-1))

    def normalize(self) -> NumericalTarget:
        """Normalize the target to the unit interval using an affine transformation.

        Raises:
            IncompatibilityError: If the target image is not bounded.

        Returns:
            The normalized target.
        """
        bounds = self.get_image()
        if not bounds.is_bounded:
            raise IncompatibilityError("Only bounded targets can be normalized.")

        return self._append_transformation(
            AffineTransformation.from_values_mapped_to_unit_interval(*bounds.to_tuple())
        )

    def abs(self) -> NumericalTarget:
        """Apply an absolute transformation to the target.

        Returns:
            The target with applied absolute transformation.
        """
        return self._append_transformation(AbsoluteTransformation())

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

    def _hold_output(
        self, abscissa: float, direction: MatchMode | str, /
    ) -> NumericalTarget:
        """Hold the target value beyond a certain abscissa value."""
        direction = MatchMode(direction)

        return evolve(  # type: ignore[call-arg]
            self, transformation=self.transformation._hold_output(abscissa, direction)
        )

    def hold_output_left_from(self, abscissa: float, /) -> NumericalTarget:
        """Hold the output of the target left from a given abscissa value."""
        return evolve(  # type: ignore[call-arg]
            self, transformation=self.transformation.hold_output_left_from(abscissa)
        )

    def hold_output_right_from(self, abscissa: float, /) -> NumericalTarget:
        """Hold the output of the target right from a given abscissa value."""
        return evolve(  # type: ignore[call-arg]
            self, transformation=self.transformation.hold_output_right_from(abscissa)
        )

    def hold_output_outside(
        self, interval: ConvertibleToInterval, /
    ) -> NumericalTarget:
        """Hold the output of the target outside a given interval."""
        return evolve(  # type: ignore[call-arg]
            self, transformation=self.transformation.hold_output_outside(interval)
        )

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

    def power(self, exponent: int) -> NumericalTarget:
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

        from baybe.utils.dataframe import to_tensor

        return pd.Series(
            self.transformation(to_tensor(series)),
            index=series.index,
            name=series.name,
        )

    @override
    def summary(self):
        return dict(
            Type=self.__class__.__name__,
            Name=self.name,
            Transformation=self.transformation,
            Minimize=self.minimize,
        )


# >>> Deprecation >>> #


@converter.register_structure_hook
def _(dct, cls) -> NumericalTarget:
    if "mode" in dct:
        return NumericalTarget(*dct)
    return select_constructor_hook(dct, cls)


_hook = converter.get_structure_hook(NumericalTarget)


@converter.register_structure_hook
def _structure_legacy_target_arguments(x: dict[str, Any], _) -> NumericalTarget:
    """Accept legacy target argument for backward compatibility."""
    x.pop("type", None)
    try:
        return _hook(x, _)
    except Exception:
        return NumericalTarget(**x)  # type: ignore[return-value]


# <<< Deprecation <<< #


gc.collect()
