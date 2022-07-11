# pylint: disable=too-few-public-methods
"""
Strategies for Design of Experiments (DoE).
"""

from abc import ABC, abstractmethod
from typing import Optional, Type, Union

import numpy as np
import pandas as pd
from botorch.acquisition import AcquisitionFunction, ExpectedImprovement

from baybe.recommender import MarginalRankingRecommender, Recommender
from baybe.surrogate import GaussianProcessModel, SurrogateModel


class InitialStrategy(ABC):
    """Abstract base class for all initial design strategies."""

    @abstractmethod
    def recommend(self, candidates: pd.DataFrame, batch_quantity: int = 1) -> pd.Index:
        """
        Selects a first subset of points from the given candidates.

        Parameters
        ----------
        candidates : pd.DataFrame
            The features of all candidate experiments that could be conducted first.
        batch_quantity : int (default = 1)
            The number of experiments to be conducted in parallel.

        Returns
        -------
        The DataFrame indices of the specific experiments selected by the strategy.
        """


class RandomInitialStrategy(InitialStrategy):
    """An initial strategy that selects the candidates at random."""

    def recommend(self, candidates: pd.DataFrame, batch_quantity: int = 1) -> pd.Index:
        """Uniform random selection of candidates."""
        return pd.Index(
            np.random.choice(candidates.index, batch_quantity, replace=False)
        )


# define allowed quick selections
SURROGATE_MODELS = {"gp": GaussianProcessModel}
ACQUISITION_FUNCTIONS = {"ei": ExpectedImprovement}
INITIAL_STRATEGIES = {"random": RandomInitialStrategy}
RECOMMENDERS = {"ranking": MarginalRankingRecommender}


class Strategy:
    """Abstract base class for all DoE strategies."""

    def __init__(
        self,
        surrogate_model_cls: Union[str, Type[SurrogateModel]] = "gp",
        acquisition_function: Union[str, Type[AcquisitionFunction]] = "ei",
        initial_strategy: Union[str, InitialStrategy] = "random",
        recommender_cls: Union[str, Type[Recommender]] = "ranking",
    ):
        # process input arguments
        self.surrogate_model_cls = self._select_surrogate_model_cls(surrogate_model_cls)
        self.acquisition_function_cls = self._select_acquisition_function_cls(
            acquisition_function
        )
        self.initial_strategy = self._select_initial_strategy(initial_strategy)
        self.recommender = self._select_recommender_cls(recommender_cls)

        # declare remaining members
        self.surrogate_model: Optional[SurrogateModel] = None
        self._best_f: Optional[float] = None
        self._use_initial_strategy: bool = False

    def fit(self, train_x: pd.DataFrame, train_y: pd.DataFrame):
        """
        Uses the given data to train a fresh surrogate model instance for the DoE
        strategy. If available, existing training data will be overwritten.

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

    def recommend(self, candidates: pd.DataFrame, batch_quantity: int = 1) -> pd.Index:
        """
        Recommends the next experiments to be conducted.

        Parameters
        ----------
        candidates : pd.DataFrame
            The features of all candidate experiments that could be conducted next.
        batch_quantity : int (default = 1)
            The number of experiments to be conducted in parallel.

        Returns
        -------
        The DataFrame indices of the specific experiments selected by the DoE strategy.
        """
        if batch_quantity > 1:
            raise NotImplementedError(
                "Batch sizes larger than 1 are not yet supported."
            )

        # if no training data exists, apply the strategy for initial recommendations
        if self._use_initial_strategy:
            return self.initial_strategy.recommend(candidates, batch_quantity)

        # construct the acquisition function
        # TODO: the current approach only works for gpytorch GP surrogate models
        #   (for other surrogate models, some wrapper is required)
        acqf = self.acquisition_function_cls(self.surrogate_model.model, self._best_f)

        # select the next experiments using the given recommender approach
        recommender = self.recommender(acqf)
        idxs = recommender.recommend(candidates, batch_quantity)

        return idxs

    @staticmethod
    def _select_surrogate_model_cls(
        surrogate_model_cls: Union[str, Type[SurrogateModel]]
    ) -> Type[SurrogateModel]:
        try:
            if (not isinstance(surrogate_model_cls, str)) and (
                issubclass(surrogate_model_cls, SurrogateModel)
            ):
                return surrogate_model_cls
            return SURROGATE_MODELS[surrogate_model_cls]
        except Exception as exc:
            raise ValueError(
                f"Undefined surrogate model '{surrogate_model_cls}'. "
                f"Must be one of {set(SURROGATE_MODELS.keys())} "
                f"or a proper 'SurrogateModel' subclass."
            ) from exc

    @staticmethod
    def _select_acquisition_function_cls(
        acquisition_function_cls: Union[str, Type[AcquisitionFunction]]
    ) -> Type[AcquisitionFunction]:
        try:
            if (not isinstance(acquisition_function_cls, str)) and (
                issubclass(acquisition_function_cls, AcquisitionFunction)
            ):
                return acquisition_function_cls
            return ACQUISITION_FUNCTIONS[acquisition_function_cls]
        except Exception as exc:
            raise ValueError(
                f"Undefined acquisition function '{acquisition_function_cls}'. "
                f"Must be one of {set(ACQUISITION_FUNCTIONS.keys())} "
                f"or a proper 'AcquisitionFunction' subclass."
            ) from exc

    @staticmethod
    def _select_initial_strategy(
        initial_strategy: Union[str, InitialStrategy]
    ) -> InitialStrategy:
        try:
            if isinstance(initial_strategy, InitialStrategy):
                return initial_strategy
            return INITIAL_STRATEGIES[initial_strategy]()
        except Exception as exc:
            raise ValueError(
                f"Undefined initial strategy '{initial_strategy}'. "
                f"Must be one of {set(INITIAL_STRATEGIES.keys())} "
                f"or a proper 'InitialStrategy' object."
            ) from exc

    @staticmethod
    def _select_recommender_cls(
        recommender_cls: Union[str, Type[Recommender]]
    ) -> Type[Recommender]:
        try:
            if (not isinstance(recommender_cls, str)) and (
                issubclass(recommender_cls, Recommender)
            ):
                return recommender_cls
            return RECOMMENDERS[recommender_cls]
        except Exception as exc:
            raise ValueError(
                f"Undefined recommender '{recommender_cls}'. "
                f"Must be one of {set(RECOMMENDERS.keys())} "
                f"or a proper 'Recommender' subclass."
            ) from exc
