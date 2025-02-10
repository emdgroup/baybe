"""Gaussian process surrogate model."""

import torch

from baybe.surrogates.base import Surrogate


class GaussianProcessSurrogate(Surrogate):
    """Gaussian process surrogate model."""

    def __init__(self, device: torch.device | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.device = device
        # later, after creating your model:
        if self.device is not None:
            self.model = self.model.to(self.device)
