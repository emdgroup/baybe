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
        *,
        n_projections: int,
        projection_matrix: _ConvertibleToTensor | None = None,
        learn_projection: bool = False,
    ):
        super().__init__()

        self.base_kernel = base_kernel
        self.n_projections = n_projections
        self.learn_projection = learn_projection

        if projection_matrix is not None:
            self._set_projection_matrix(
                torch.as_tensor(projection_matrix, dtype=DTypeFloatTorch)
            )

    @staticmethod
    def _make_projection_matrix(n_input_dims: int, n_projections) -> Tensor:
        """Generate a random Gaussian projection matrix."""
        return torch.randn(n_projections, n_input_dims, dtype=DTypeFloatTorch).div(
            n_projections**0.5
        )

    def _set_projection_matrix(self, matrix: Tensor, /) -> None:
        """Set the projection matrix as a parameter or buffer."""
        type_ = torch.nn.Parameter if self.learn_projection else torch.nn.Buffer
        self.projection_matrix = type_(matrix)

    def forward(self, x1: Tensor, x2: Tensor, **kwargs):
        """Apply the base kernel to the projected input tensors."""
        if not hasattr(self, "projection_matrix"):
            self._set_projection_matrix(
                self._make_projection_matrix(self.n_projections, x1.size(-1))
            )

        x1_proj = x1 @ self.projection_matrix
        x2_proj = x2 @ self.projection_matrix
        return self.base_kernel(x1_proj, x2_proj, **kwargs)
