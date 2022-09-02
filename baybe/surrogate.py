# pylint: disable=too-few-public-methods
"""
Surrogate models, such as Gaussian processes, random forests, etc.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Type

import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim.fit import fit_gpytorch_torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from .utils import to_tensor


class SurrogateModel(ABC):
    """Abstract base class for all surrogate models."""

    # TODO: to support other models than GPs, an interface to botorch's acquisition
    #  functions must be created (e.g. via a dedicated 'predict' method)

    type: str
    SUBCLASSES: Dict[str, Type[SurrogateModel]] = {}

    @abstractmethod
    def posterior(self, candidates: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluates the surrogate model at the given candidate points."""

    def fit(self, train_x: pd.DataFrame, train_y: pd.DataFrame) -> None:
        """Trains the surrogate model on the provided data."""

    @classmethod
    def __init_subclass__(cls, **kwargs):
        """Registers new subclasses dynamically."""
        super().__init_subclass__(**kwargs)
        cls.SUBCLASSES[cls.type] = cls


class GaussianProcessModel(SurrogateModel):
    """A Gaussian process surrogate model."""

    type = "GP"

    def __init__(self, searchspace: pd.DataFrame):
        self.model: Optional[SingleTaskGP] = None
        self.searchspace = searchspace

    def posterior(self, candidates: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """See base class."""
        posterior = self.model.posterior(candidates)

        # use numpy output type to remain consistent with the function signature
        # TODO: change signature to torch when implementing continuous parameters
        mean = posterior.mvn.mean.detach().numpy()
        covar = posterior.mvn.covariance_matrix.detach().numpy()

        return mean, covar

    def fit(self, train_x: pd.DataFrame, train_y: pd.DataFrame) -> None:
        """See base class."""

        # validate input
        if len(train_x) == 0:
            raise ValueError("The training data set must be non-empty.")
        if train_y.shape[1] != 1:
            raise NotImplementedError("The model currently supports only one target.")

        # get the input bounds from the search space
        searchspace = to_tensor(self.searchspace)
        bounds = torch.vstack(
            [torch.min(searchspace, dim=0)[0], torch.max(searchspace, dim=0)[0]]
        )
        # TODO: use target value bounds when explicitly provided

        # define the input and outcome transforms
        input_transform = Normalize(train_x.shape[1], bounds=bounds)
        outcome_transform = Standardize(train_y.shape[1])

        # convert dataframes to tensors for the GP model
        train_x, train_y = to_tensor(train_x, train_y)

        # construct and fit the Gaussian process
        self.model = SingleTaskGP(
            train_x,
            train_y,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(mll, optimizer=fit_gpytorch_torch, options={"disp": False})
