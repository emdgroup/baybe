from __future__ import annotations

from typing import Any
from torch import Tensor

from copy import deepcopy

import gpytorch
import torch
from botorch.models import SingleTaskGP


class PriorMean(gpytorch.means.Mean):
    """GPyTorch mean module using a trained GP as prior mean.

    This mean module wraps a trained Gaussian Process and uses its predictions
    as the mean function for another GP.
    """

    def __init__(
        self, gp: SingleTaskGP, batch_shape: torch.Size = torch.Size(), **kwargs: Any
    ) -> None:
        """Initialize the GP-based mean module.

        Args:
            gp: Trained Gaussian Process to use as mean function.
            batch_shape: Batch shape for the mean module.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        # See https://github.com/cornellius-gp/gpytorch/issues/743
        self.gp: SingleTaskGP = deepcopy(gp)
        self.batch_shape: torch.Size = batch_shape
        for param in self.gp.parameters():
            param.requires_grad = False

    def reset_gp(self) -> None:
        """Reset the GP to evaluation mode for prediction."""
        self.gp.eval()
        self.gp.likelihood.eval()

    def forward(self, input: Tensor) -> Tensor:
        """Compute the mean function using the wrapped GP.

        Args:
            input: Input tensor for which to compute the mean.

        Returns:
            Mean predictions from the wrapped GP.
        """
        self.reset_gp()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            with gpytorch.settings.detach_test_caches(False):
                mean = self.gp(input).mean.detach()
        mean = mean.reshape(torch.broadcast_shapes(self.batch_shape, input.shape[:-1]))
        return mean


class PriorKernel(gpytorch.kernels.Kernel):
    """GPyTorch kernel module wrapping a pre-trained kernel.

    This kernel module wraps a trained kernel and uses it as a fixed kernel
    component in another GP. The wrapped kernel's parameters are frozen.
    """

    def __init__(self, kernel, **kwargs):
        """Initialize the kernel wrapper.

        Args:
            kernel: Pre-trained kernel to wrap.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        # See https://github.com/cornellius-gp/gpytorch/issues/743
        self.base_kernel = deepcopy(kernel)
        for param in self.base_kernel.parameters():
            param.requires_grad = False

    def reset(self):
        """Reset the wrapped kernel to evaluation mode."""
        self.base_kernel.eval()

    def forward(self, x1, x2, **params):
        """Compute kernel matrix using the wrapped kernel.

        Args:
            x1: First set of input points.
            x2: Second set of input points.
            **params: Additional kernel parameters.

        Returns:
            Kernel matrix computed by the wrapped kernel.
        """
        self.reset()
        with gpytorch.settings.fast_pred_var():
            with gpytorch.settings.detach_test_caches(False):
                k = self.base_kernel.forward(x1, x2, **params)
        return k