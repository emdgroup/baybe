"""GPyTorch kernel implementations."""

from typing import Any

import torch
from gpytorch.kernels import Kernel
from torch import Tensor

from baybe.utils.torch import DTypeFloatTorch

_ConvertibleToTensor = Any
"""A type alias for objects convertible to tensors."""


class ProjectionKernel(Kernel):
    """GPyTorch implementation of :class:`baybe.kernels.composite.ProjectionKernel`."""

    def __init__(
        self,
        base_kernel: Kernel,
        projection_matrix: _ConvertibleToTensor,
        *,
        learn_projection: bool = False,
    ):
        super().__init__()

        self.base_kernel = base_kernel
        self.learn_projection = learn_projection

        matrix = torch.tensor(projection_matrix, dtype=DTypeFloatTorch)
        if self.learn_projection:
            self.register_parameter("projection_matrix", torch.nn.Parameter(matrix))
        else:
            self.register_buffer("projection_matrix", matrix)

    def forward(self, x1: Tensor, x2: Tensor, **kwargs):
        """Apply the base kernel to the projected input tensors."""
        x1_proj = x1 @ self.projection_matrix
        x2_proj = x2 @ self.projection_matrix
        return self.base_kernel(x1_proj, x2_proj, **kwargs)
