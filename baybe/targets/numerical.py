"""Numerical targets."""

import gc
import warnings

import pandas as pd
import torch
from attrs import define, field
from typing_extensions import override

from baybe.serialization import SerialMixin
from baybe.targets.base import Target
from baybe.targets.transforms import TransformationProtocol


@define(frozen=True)
class NumericalTarget(Target, SerialMixin):
    """Class for numerical targets."""

    transformation: TransformationProtocol | None = field(default=None)
    """An optional target transformation."""

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
