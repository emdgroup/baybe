"""
Scaler class for input/output scaling
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Type

import pandas as pd
import torch
from torch import Tensor
from .utils import to_tensor


class Scaler(ABC):
    """Abstract base class for all surrogate models."""

    type: str
    SUBCLASSES: Dict[str, Type[Scaler]] = {}

    def __init__(self, searchspace: pd.DataFrame):
        self.searchspace = searchspace
        self.fitted = False
        self.scaleX = None
        self.scaleY = None
        self.unscaleX = None
        self.unscaleY = None
        self.unscaleS = None

    @abstractmethod
    def fit(self, X: Tensor, Y: Tensor) -> None:
        """
        Creates scaling functions based on the input data
        """

    @abstractmethod
    def transform(self, X: Tensor, Y: Tensor) -> Tuple[Tensor, Tensor]:
        """Transforms an input"""
    
    @abstractmethod
    def untransform(self, M: Tensor, S: Tensor) -> Tuple[Tensor, Tensor]:
        """Untransforms an output"""

    @classmethod
    def __init_subclass__(cls, **kwargs):
        """Registers new subclasses dynamically."""
        super().__init_subclass__(**kwargs)
        cls.SUBCLASSES[cls.type] = cls


class DefaultScaler(Scaler):
    """A simple scaler with X normalized, Y standardized"""
    
    type = "DEFAULT"

    def fit(self, X: Tensor, Y: Tensor) -> None:
        """See base class."""
        # Get searchspace
        searchspace = to_tensor(self.searchspace)
        # Find bounds of X
        bounds = torch.vstack(
            [torch.min(searchspace, dim=0)[0], torch.max(searchspace, dim=0)[0]]
        )

        
        # Find mean, std of Y
        mean = torch.mean(Y, dim=0)
        std = torch.std(Y, dim=0)

        # Define scaling functions
        self.scaleX = lambda X: (X-bounds[0])/(bounds[1]-bounds[0])
        self.scaleY = lambda Y: (Y-mean)/std

        self.unscaleX = lambda X: X*(bounds[1]-bounds[0]) + bounds[0]
        self.unscaleY = lambda Y: Y*std + mean

        self.unscaleM = lambda M: M*std + mean
        self.unscaleS = lambda S: S*std**2

        self.fitted = True


    def transform(self, X: Tensor, Y: Tensor = None) -> Tuple[Tensor, Tensor]:
        """See base class."""

        # Ensure scaler has been fitted
        if not self.fitted:
            raise RuntimeError("Scaler object must be fitted first")

        if not Y == None:
            # Training (fit) mode
            return (self.scaleX(X), self.scaleY(Y))
        else:
            # Predict (posterior) mode
            
            # Flatten t-batch
            flattened = X.flatten(end_dim=-3)
            
            # Get scaled values
            scaled = [self.scaleX(t).unsqueeze(1) for t in flattened.unbind(dim=-2)]

            # Combine scaled values
            scaled = torch.cat(tuple(scaled), dim=-1).reshape(X.shape)

            return (scaled, None)
    
    def untransform(self, M: Tensor, S: Tensor = None) -> Tuple[Tensor, Tensor]:
        # Ensure scaler has been fitted
        if not self.fitted:
            raise RuntimeError("Scaler object must be fitted first")
        return (self.unscaleM(M), self.unscaleS(S))
