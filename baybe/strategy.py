"""
Design-of-Experiment strategies, such as Gaussian processes, random forests, etc.
"""

from abc import ABC
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
import torch
from botorch.acquisition import ExpectedImprovement
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim.fit import fit_gpytorch_torch
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor


class Strategy(ABC):
    """Abstract base class for all Design-of-Experiments (DoE) strategies."""

    def set_training_data(self, train_x: pd.DataFrame, train_y: pd.DataFrame):
        """
        Sets the training data for the DoE strategy and trains a fresh model instance.
        If available, existing training data will be overwritten.

        Parameters
        ----------
        train_x : pd.DataFrame
            The features of the conducted experiments.
        train_y : pd.DataFrame
            The corresponding response values.
        """

    def recommend(self, candidates: pd.DataFrame, batch_size: int = 1) -> pd.Index:
        """
        Recommends the next experiments to be conducted.

        Parameters
        ----------
        candidates : pd.DataFrame
            The features of all candidate experiments that could be conducted next.
        batch_size : int (default = 1)
            The number of experiments to be conducted in parallel.

        Returns
        -------
        The DataFrame indices of the specific experiments selected by the DoE strategy.
        """


class GaussianProcessStrategy(ABC):
    """A DoE strategy using a Gaussian Process (GP) surrogate model."""

    def __init__(self):
        self.model: Optional[SingleTaskGP] = None
        self.best_f: Optional[float] = None

    def set_training_data(self, train_x: pd.DataFrame, train_y: pd.DataFrame):
        """See base class."""
        # TODO: assert correct input and target scaling

        # convert dataframes to tensors for the GP model
        train_x, train_y = self._to_tensor(train_x, train_y)

        # for an empty training set, reset the model
        if len(train_x) == 0:
            self.best_f = None
            self.model = None
            return

        # initialize the GP model and train it
        self.model = SingleTaskGP(train_x, train_y)
        self._fit()

        # store the current best response value
        self.best_f = train_y.min()

    def recommend(self, candidates: pd.DataFrame, batch_size: int = 1) -> pd.Index:
        """See base class."""

        if batch_size > 1:
            raise NotImplementedError(
                "Batch sizes larger than 1 are not yet supported."
            )

        # if no training data exists, apply the strategy for initial recommendations
        if self.model is None:
            # TODO: add more options
            return np.random.choice(candidates.index)

        # prepare the candidates in t-batches
        candidates_tensor = self._to_tensor(candidates).unsqueeze(1)

        # construct and evaluate the acquisition function
        # TODO: add more options
        acqf = ExpectedImprovement(self.model, self.best_f)
        acqf_values = acqf(candidates_tensor)

        # find the minimizer and extract the corresponding dataframe index
        # TODO: use botorch's built-in methods
        #   (problem: they do not return the indices but the candidate points)
        min_iloc = torch.argmin(acqf_values).item()
        min_loc = candidates.index[min_iloc]

        return min_loc

    @staticmethod
    def _to_tensor(*dfs: pd.DataFrame) -> Union[Tensor, Iterable[Tensor]]:
        """Converts a given set of dataframes into tensors (dropping all indices)."""
        out = (torch.from_numpy(df.values).to(torch.float32) for df in dfs)
        if len(dfs) == 1:
            out = next(out)
        return out

    def _fit(self):
        """Fits the GP model with the current training data."""
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(mll, optimizer=fit_gpytorch_torch, options={"disp": False})
