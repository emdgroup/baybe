"""
Scaler class for input/output scaling
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Type

import pandas as pd
import torch
from torch import Tensor

from .utils import to_tensor


def _smooth_y(y: Tensor):
    """
    Helper function to smooth y to avoid variance nearing zero (numerical instability)
    """
    # Add small (random) tensor to y
    amplitude = 1e-3
    return y + amplitude * torch.randn(y.shape)


class Scaler(ABC):
    """Abstract base class for all surrogate models."""

    type: str
    SUBCLASSES: Dict[str, Type[Scaler]] = {}

    def __init__(self, searchspace: pd.DataFrame):
        self.searchspace = searchspace
        self.fitted = False
        self.scale_x = None
        self.scale_y = None
        self.unscale_x = None
        self.unscale_y = None
        self.unscale_m = None
        self.unscale_s = None

    @abstractmethod
    def fit_transform(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Creates scaling functions based on the input data and transforms input
        """

    @abstractmethod
    def transform(self, x: Tensor) -> Tensor:
        """Transforms an input"""

    @abstractmethod
    def untransform(self, m_tensor: Tensor, s_tensor: Tensor) -> Tuple[Tensor, Tensor]:
        """Untransforms an output"""

    @classmethod
    def __init_subclass__(cls, **kwargs):
        """Registers new subclasses dynamically."""
        super().__init_subclass__(**kwargs)
        cls.SUBCLASSES[cls.type] = cls


class DefaultScaler(Scaler):
    """A simple scaler with x normalized, y standardized"""

    type = "DEFAULT"

    def fit_transform(self, x: Tensor, y: Tensor) -> None:
        """See base class."""
        # Get searchspace
        searchspace = to_tensor(self.searchspace)
        # Find bounds of x
        bounds = torch.vstack(
            [torch.min(searchspace, dim=0)[0], torch.max(searchspace, dim=0)[0]]
        )

        # Find mean, std of y
        mean = torch.mean(y, dim=0)
        std = torch.std(y, dim=0)

        # Add noise to std if very small
        if std < 1e-6:
            y = _smooth_y(y)
            std = torch.std(y, dim=0)

        # Define scaling functions
        self.scale_x = lambda l: (l - bounds[0]) / (bounds[1] - bounds[0])
        self.scale_y = lambda l: (l - mean) / std

        self.unscale_x = lambda l: l * (bounds[1] - bounds[0]) + bounds[0]
        self.unscale_y = lambda l: l * std + mean

        self.unscale_m = lambda l: l * std + mean
        self.unscale_s = lambda l: l * std**2

        self.fitted = True

        return (self.scale_x(x), self.scale_y(y))

    def transform(self, x: Tensor) -> Tensor:
        """See base class."""

        # Ensure scaler has been fitted
        if not self.fitted:
            raise RuntimeError("Scaler object must be fitted first")

        # Check if batching is needed
        if len(x.shape) > 2:
            # Predict (posterior) mode
            # Flatten t-batch
            flattened = x.flatten(end_dim=-3)

            # Get scaled values
            scaled = [self.scale_x(t).unsqueeze(1) for t in flattened.unbind(dim=-2)]

            # Combine scaled values
            scaled = torch.cat(tuple(scaled), dim=-1).reshape(x.shape)
        else:
            scaled = self.scale_x(x)

        return scaled

    def untransform(
        self, m_tensor: Tensor, s_tensor: Tensor = None
    ) -> Tuple[Tensor, Tensor]:
        # Ensure scaler has been fitted
        if not self.fitted:
            raise RuntimeError("Scaler object must be fitted first")
        return (self.unscale_m(m_tensor), self.unscale_s(s_tensor))
