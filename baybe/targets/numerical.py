"""Numerical targets."""

from __future__ import annotations

import gc
import warnings
from collections.abc import Iterable

import pandas as pd
import torch
from attrs import define, field
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
)
from baybe.utils.interval import Interval


@define(frozen=True)
class NumericalTarget(Target, SerialMixin):
    """Class for numerical targets."""

    transformation: TransformationProtocol | None = field(default=None)
    """An optional target transformation."""

    @classmethod
    def match_triangular(cls, name: str, cutoffs: Iterable[float]) -> NumericalTarget:
        interval = Interval.create(cutoffs)
        return NumericalTarget(
            name,
            ChainedTransformation(
                AffineTransformation.from_unit_interval(
                    interval.center, interval.upper
                ),
                AbsoluteTransformation(),
                AffineTransformation(factor=-1, shift=1),
                ClampingTransformation(min=0, max=1),
            ),
        )

    @classmethod
    def match_bell(cls, name: str, center: float, width: float) -> NumericalTarget:
        return NumericalTarget(name, BellTransformation(center, width))

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
            return pd.Series(
                self.transformation.transform(torch.from_numpy(series.to_numpy())),
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
