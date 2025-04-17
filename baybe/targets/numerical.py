"""Numerical targets."""

from __future__ import annotations

import gc
import warnings
from collections.abc import Iterable
from typing import Any

import pandas as pd
from attrs import define, field
from attrs.converters import optional
from attrs.validators import instance_of
from typing_extensions import override

from baybe.serialization import SerialMixin, converter
from baybe.targets._deprecated import TargetMode, TargetTransformation
from baybe.targets.base import Target
from baybe.targets.transforms import (
    AbsoluteTransformation,
    AffineTransformation,
    BellTransformation,
    ChainedTransformation,
    ClampingTransformation,
    Transformation,
    TransformationProtocol,
    convert_transformation,
)
from baybe.utils.interval import Interval


@define(frozen=True, init=False)
class NumericalTarget(Target, SerialMixin):
    """Class for numerical targets."""

    transformation: TransformationProtocol | None = field(
        default=None, converter=optional(convert_transformation)
    )
    """An optional target transformation."""

    minimize: bool = field(default=False, validator=instance_of(bool), kw_only=True)

    def __init__(  # noqa: DOC301
        self,
        name: str,
        transformation_: TransformationProtocol  # underscore to avoid name collision
        | None = None,
        *args,
        minimize: bool = False,
        **kwargs,
    ):
        """Translate legacy target specifications."""
        # Check if legacy arguments are provided
        if (
            not (
                isinstance(transformation_, (Transformation, type(None)))
                or callable(transformation_)
            )
            or args
            or kwargs
        ):
            # Map legacy arguments to legacy constructor parameter names
            kw: dict[str, Any] = {
                "name": name,
                "mode": None,
                "bounds": None,
                "transformation": None,
            }
            all_args = (transformation_, *args)
            if transformation_ in (
                *TargetTransformation.__members__.keys(),
                *TargetTransformation.__members__.values(),
            ):
                kw["transformation"] = transformation_
                all_args = all_args[1:]
            for k, v in zip(
                ["mode", "bounds", "transformation"],
                [v for v in all_args],  # if v is not None],
            ):
                kw[k] = v
            kw = kw | kwargs

            warnings.warn(
                "Creating numerical targets by specifying MAX/MIN/MATCH modes has been "
                "deprecated. For now, you do not need to change your code as we "
                "automatically converted your target to the new format. "
                "However, this functionality will be removed in a future version, so "
                "please familiarize yourself with the new API.",
                DeprecationWarning,
            )

            # Translate to modern API
            mode = TargetMode[kw["mode"]]
            bounds = kw["bounds"]
            if mode in (TargetMode.MAX, TargetMode.MIN):
                if bounds is None:
                    self.__attrs_init__(kw["name"], minimize=mode == TargetMode.MIN)
                else:
                    self.__attrs_init__(
                        kw["name"],
                        NumericalTarget.clamped_affine(
                            "dummy", cutoffs=bounds, descending=mode == TargetMode.MIN
                        ).transformation,
                    )
            else:
                if kw["transformation"] == "BELL":
                    center = (bounds[1] + bounds[0]) / 2
                    width = (bounds[1] - bounds[0]) / 2
                    transformation = BellTransformation(center, width)
                else:
                    transformation = NumericalTarget.match_triangular(
                        "dummy", bounds
                    ).transformation
                self.__attrs_init__(kw["name"], transformation)

        else:
            self.__attrs_init__(name, transformation_, minimize=minimize)

    @classmethod
    def match_triangular(cls, name: str, cutoffs: Iterable[float]) -> NumericalTarget:
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
    def clamped_affine(
        cls, name: str, cutoffs: Iterable[float], *, descending: bool = False
    ) -> NumericalTarget:
        """Create a target that is affine in a given range and clamped to 0/1 outside.

        Args:
            name: The name of the target.
            cutoffs: The cutoff values defining the affine region.
            descending: Boolean flag indicating if the transformation is ascending
                or descending in the affine region.

        Returns:
            The target with applied clamped linear transformation.
        """
        bounds = Interval.create(cutoffs).to_tuple()
        if descending:
            bounds = bounds[::-1]
        return NumericalTarget(
            name, AffineTransformation.from_unit_interval(*bounds).clamp(0, 1)
        )

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
        if (trans := self.transformation) is not None or self.minimize:
            import torch

            if self.minimize:
                if trans is None:
                    trans = AffineTransformation(factor=-1)
                else:
                    trans = ChainedTransformation(
                        trans, AffineTransformation(factor=-1)
                    )
            assert trans is not None
            return pd.Series(
                trans(torch.from_numpy(series.to_numpy())),
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
