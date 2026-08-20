"""Base class for aggregation functions."""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from attrs import define

from baybe.serialization.mixin import SerialMixin

if TYPE_CHECKING:
    from torch import Tensor


@define(frozen=True)
class AggregationFunction(ABC, SerialMixin):
    """Abstract base class for aggregation over contexts, for generality BO."""

    @abstractmethod
    def forward(self, Y: Tensor) -> Tensor:
        """Aggregate over the last dimension.

        Args:
            Y: Tensor of shape (..., r) where r is the number of contexts.

        Returns:
            Tensor of shape (...) with the aggregated values.
        """


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
