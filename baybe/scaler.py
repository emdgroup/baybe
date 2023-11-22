"""Functionality for data scaling."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, Tuple, Type

import pandas as pd
import torch
from torch import Tensor

from baybe.utils import to_tensor

_ScaleFun = Callable[[Tensor], Tensor]


class Scaler(ABC):
    """Abstract base class for all scalers.

    Args:
        searchspace: The search space that should be scaled.
    """

    type: str
    """Class variable encoding the type of the scaler."""

    SUBCLASSES: Dict[str, Type[Scaler]] = {}
    """Class variable for all subclasses"""

    def __init__(self, searchspace: pd.DataFrame):
        self.searchspace = searchspace
        self.fitted = False
        self.scale_x: _ScaleFun
        self.scale_y: _ScaleFun
        self.unscale_x: _ScaleFun
        self.unscale_y: _ScaleFun
        self.unscale_m: _ScaleFun
        self.unscale_s: _ScaleFun

    @abstractmethod
    def fit_transform(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Fit the scaler using the given training data and transform the data.

        Args:
            x: The x-data that should be used.
            y: The y-data that should be used.

        Returns:
            The transformed data.
        """

    def transform(self, x: Tensor) -> Tensor:
        """Scale a given input.

        Args:
            x: The given input.

        Returns:
            The scaled input.

        Raises:
            RuntimeError: If the scaler is not fitted first.
        """
        if not self.fitted:
            raise RuntimeError("Scaler object must be fitted first.")
        return self.scale_x(x)

    def untransform(self, mean: Tensor, variance: Tensor) -> Tuple[Tensor, Tensor]:
        """Transform mean values and variances back to the original domain.

        Args:
            mean: The given mean values.
            variance: The given variances.

        Returns:
            The "un-transformed" means and variances.

        Raises:
            RuntimeError: If the scaler object is not fitted first.
        """
        if not self.fitted:
            raise RuntimeError("Scaler object must be fitted first.")
        return self.unscale_m(mean), self.unscale_s(variance)

    @classmethod
    def __init_subclass__(cls, **kwargs):
        """Register new subclasses dynamically."""
        super().__init_subclass__(**kwargs)
        cls.SUBCLASSES[cls.type] = cls


class DefaultScaler(Scaler):
    """A scaler that normalizes inputs to the unit cube and standardizes targets."""

    type = "DEFAULT"
    # See base class.

    def fit_transform(  # noqa: D102
        self, x: Tensor, y: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # See base class.

        # Get the searchspace boundaries
        searchspace = to_tensor(self.searchspace)
        bounds = torch.vstack(
            [torch.min(searchspace, dim=0)[0], torch.max(searchspace, dim=0)[0]]
        )

        # Compute the mean and standard deviation of the training targets
        mean = torch.mean(y, dim=0)
        std = torch.std(y, dim=0)

        # Functions for input and target scaling
        self.scale_x = lambda x: (x - bounds[0]) / (bounds[1] - bounds[0])
        self.scale_y = lambda x: (x - mean) / std

        # Functions for inverse input and target scaling
        self.unscale_x = lambda x: x * (bounds[1] - bounds[0]) + bounds[0]
        self.unscale_y = lambda x: x * std + mean

        # Functions for inverse mean and variance scaling
        self.unscale_m = lambda x: x * std + mean
        self.unscale_s = lambda x: x * std**2

        # Flag that the scaler has been fitted
        self.fitted = True

        return self.scale_x(x), self.scale_y(y)
