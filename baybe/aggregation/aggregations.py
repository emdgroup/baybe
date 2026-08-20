"""Aggregation functions."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

from attrs import define, field
from attrs.validators import gt
from typing_extensions import override

from baybe.aggregation.base import AggregationFunction
from baybe.utils.validation import finite_float

if TYPE_CHECKING:
    from torch import Tensor


@define(frozen=True)
class MeanAggregation(AggregationFunction):
    """Mean over contexts."""

    @override
    def forward(self, Y: Tensor) -> Tensor:
        """See base class."""
        return Y.mean(dim=-1)


@define(frozen=True)
class MinAggregation(AggregationFunction):
    """Minimum over contexts (worst-case generality)."""

    @override
    def forward(self, Y: Tensor) -> Tensor:
        """See base class."""
        return Y.min(dim=-1).values


@define(frozen=True)
class SigmoidAggregation(AggregationFunction):
    """Fraction of contexts above threshold."""

    threshold: float = field(validator=[finite_float])
    """The threshold above which contexts count as successful."""

    steepness: float = field(default=50.0, validator=[finite_float, gt(0.0)])
    """Sigmoid steepness for differentiable threshold approximation."""

    @override
    def forward(self, Y: Tensor) -> Tensor:
        """See base class."""
        import torch

        sigmoid_values = torch.sigmoid(self.steepness * (Y - self.threshold))
        return sigmoid_values.mean(dim=-1)


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
