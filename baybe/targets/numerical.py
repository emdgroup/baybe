"""Numerical targets."""

from __future__ import annotations

import gc
import warnings
from collections.abc import Iterable

import pandas as pd
from attrs import define, field
from attrs.converters import optional
from attrs.validators import instance_of
from typing_extensions import override

from baybe.serialization import SerialMixin
from baybe.targets.base import Target
from baybe.targets.transforms import (
    AbsoluteTransformation,
    AffineTransformation,
    BellTransformation,
    ChainedTransformation,
    ClampingTransformation,
    TransformationProtocol,
    convert_transformation,
)
from baybe.utils.interval import Interval


@define(frozen=True)
class NumericalTarget(Target, SerialMixin):
    """Class for numerical targets."""

    transformation: TransformationProtocol | None = field(
        default=None, converter=optional(convert_transformation)
    )
    """An optional target transformation."""

    minimize: bool = field(default=False, validator=instance_of(bool), kw_only=True)

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
            return pd.Series(
                trans.transform(torch.from_numpy(series.to_numpy())),
                index=series.index,
                name=series.name,
            )
        else:
            return series.copy()

    @override
    def summary(self):
        raise NotImplementedError()


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
