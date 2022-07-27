# pylint: disable=too-few-public-methods
"""
Surrogate models, such as Gaussian processes, random forests, etc.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import pandas as pd
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim.fit import fit_gpytorch_torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from baybe.utils import to_tensor


class SurrogateModel(ABC):
    """Abstract base class for all surrogate models."""

    # TODO: to support other models than GPs, an interface to botorch's acquisition
    #  functions must be created (e.g. via a dedicated 'predict' method)

    TYPE: str
    SUBCLASSES: Dict[str, SurrogateModel] = {}

    @abstractmethod
    def fit(self, train_x: pd.DataFrame, train_y: pd.DataFrame):
        """Trains the surrogate model on the provided data."""

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.SUBCLASSES[cls.TYPE] = cls


class GaussianProcessModel(SurrogateModel):
    """A Gaussian process surrogate model."""

    TYPE = "GP"

    def __init__(self):
        self.model: Optional[SingleTaskGP] = None

    def fit(self, train_x: pd.DataFrame, train_y: pd.DataFrame):
        """See base class."""
        # TODO: assert correct input and target scaling

        # validate input
        if not train_x.index.equals(train_y.index):
            raise ValueError("Training inputs and targets must have the same index.")
        if len(train_x) == 0:
            raise ValueError("The training data set must be non-empty.")

        # convert dataframes to tensors for the GP model
        train_x, train_y = to_tensor(train_x, train_y)

        # initialize the GP model and train it
        self.model = SingleTaskGP(train_x, train_y)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(mll, optimizer=fit_gpytorch_torch, options={"disp": False})
