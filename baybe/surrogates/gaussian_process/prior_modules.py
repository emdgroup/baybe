"""Prior modules for Gaussian process transfer learning."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import gpytorch
import torch
from botorch.models import SingleTaskGP
from torch import Tensor


class PriorMean(gpytorch.means.Mean):
    """GPyTorch mean module using a trained GP as prior mean.

    This mean module wraps a trained Gaussian Process and uses its predictions
    as the mean function for another GP.

    Args:
        gp: Trained Gaussian Process to use as mean function.
        batch_shape: Batch shape for the mean module.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self, gp: SingleTaskGP, batch_shape: torch.Size = torch.Size(), **kwargs: Any
    ) -> None:
        super().__init__()

        # Deep copy and freeze the GP
        self.gp: SingleTaskGP = deepcopy(gp)
        self.batch_shape: torch.Size = batch_shape

        # Freeze parameters and set eval mode once
        for param in self.gp.parameters():
            param.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        """Compute the mean function using the wrapped GP.

        Args:
            x: Input tensor for which to compute the mean.

        Returns:
            Mean predictions from the wrapped GP.
        """
        self.gp.eval()
        self.gp.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            mean = self.gp(x).mean.detach()

        # Handle batch dimensions
        target_shape = torch.broadcast_shapes(self.batch_shape, x.shape[:-1])
        return mean.reshape(target_shape)
