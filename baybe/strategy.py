# pylint: disable=too-few-public-methods
"""
Strategies for Design of Experiments (DoE).
"""

from abc import ABC, abstractmethod
from typing import Optional, Type, Union

import numpy as np
import pandas as pd
import torch
from botorch.acquisition import AcquisitionFunction, ExpectedImprovement

from baybe.surrogate import GaussianProcessModel, SurrogateModel
from baybe.utils import to_tensor


class InitialStrategy(ABC):
    """Abstract base class for all initial design strategies."""

    @abstractmethod
    def recommend(self, candidates: pd.DataFrame, batch_size: int = 1) -> pd.Index:
        """
        Selects a first subset of points from the given candidates.

        Parameters
        ----------
        candidates : pd.DataFrame
            The features of all candidate experiments that could be conducted first.
        batch_size : int (default = 1)
            The number of experiments to be conducted in parallel.

        Returns
        -------
        The DataFrame indices of the specific experiments selected by the strategy.
        """


class RandomInitialStrategy(InitialStrategy):
    """An initial strategy that selects the candidates at random."""

    def recommend(self, candidates: pd.DataFrame, batch_size: int = 1) -> pd.Index:
        """Uniform random selection of candidates."""
        return pd.Index(np.random.choice(candidates.index, batch_size, replace=False))


class Strategy:
    """Abstract base class for all DoE strategies."""

    def __init__(
        self,
        surrogate_model_cls: Union[str, Type[SurrogateModel]] = "gp",
        acquisition_function: Union[str, Type[AcquisitionFunction]] = "ei",
        initial_strategy: Union[str, InitialStrategy] = "random",
    ):
        # process input arguments
        self.surrogate_model_cls = self._select_surrogate_model_cls(surrogate_model_cls)
        self.acquisition_function_cls = self._select_acquisition_function_cls(
            acquisition_function
        )
        self.initial_strategy = self._select_initial_strategy(initial_strategy)

        # declare remaining members
        self.surrogate_model: Optional[SurrogateModel] = None
        self._best_f: Optional[float] = None
        self._use_initial_strategy: bool = False

    def set_training_data(self, train_x: pd.DataFrame, train_y: pd.DataFrame):
        """
        Sets the training data for the DoE strategy and trains a fresh surrogate
        model instance. If available, existing training data will be overwritten.

        Parameters
        ----------
        train_x : pd.DataFrame
            The features of the conducted experiments.
        train_y : pd.DataFrame
            The corresponding response values.
        """
        self._use_initial_strategy = len(train_x) == 0
        if not self._use_initial_strategy:
            self.surrogate_model = self.surrogate_model_cls()
            self.surrogate_model.fit(train_x, train_y)
            self._best_f = train_y.min()

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
        if batch_size > 1:
            raise NotImplementedError(
                "Batch sizes larger than 1 are not yet supported."
            )

        # if no training data exists, apply the strategy for initial recommendations
        if self._use_initial_strategy:
            return self.initial_strategy.recommend(candidates, batch_size)

        # prepare the candidates in t-batches
        candidates_tensor = to_tensor(candidates).unsqueeze(1)

        # construct and evaluate the acquisition function
        # TODO: the current approach only works for gpytorch GP surrogate models
        #   (for other surrogate models, some wrapper is required)
        acqf = self.acquisition_function_cls(self.surrogate_model.model, self._best_f)
        acqf_values = acqf(candidates_tensor)

        # find the minimizer and extract the corresponding dataframe index
        # TODO: use botorch's built-in methods
        #   (problem: they do not return the indices but the candidate points)
        min_iloc = torch.argmin(acqf_values).item()
        min_loc = candidates.index[[min_iloc]]

        return min_loc

    @staticmethod
    def _select_surrogate_model_cls(
        surrogate_model_cls: Union[str, Type[SurrogateModel]]
    ) -> Type[SurrogateModel]:
        if (not isinstance(surrogate_model_cls, str)) and (
            issubclass(surrogate_model_cls, SurrogateModel)
        ):
            return surrogate_model_cls
        if surrogate_model_cls.lower() == "gp":
            return GaussianProcessModel
        raise ValueError("Undefined surrogate model class.")

    @staticmethod
    def _select_acquisition_function_cls(
        acquisition_function_cls: Union[str, Type[AcquisitionFunction]]
    ) -> Type[AcquisitionFunction]:
        if (not isinstance(acquisition_function_cls, str)) and (
            issubclass(acquisition_function_cls, AcquisitionFunction)
        ):
            return acquisition_function_cls
        if acquisition_function_cls.lower() == "ei":
            return ExpectedImprovement
        raise ValueError("Undefined acquisition function.")

    @staticmethod
    def _select_initial_strategy(
        initial_strategy: Union[str, InitialStrategy]
    ) -> InitialStrategy:
        if isinstance(initial_strategy, InitialStrategy):
            return initial_strategy
        if initial_strategy.lower() == "random":
            return RandomInitialStrategy()
        raise ValueError("Undefined initial strategy.")
