# pylint: disable=too-few-public-methods
"""
Strategies for Design of Experiments (DoE).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Literal, Optional, Type, Union

import numpy as np
import pandas as pd
from botorch.acquisition import AcquisitionFunction, ExpectedImprovement
from pydantic import BaseModel, Extra, validator

from baybe.recommender import Recommender
from baybe.surrogate import SurrogateModel
from baybe.utils import check_if_in


class InitialStrategy(ABC):
    """Abstract base class for all initial design strategies."""

    TYPE: str
    SUBCLASSES: Dict[str, InitialStrategy] = {}

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

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.SUBCLASSES[cls.TYPE] = cls


class RandomInitialStrategy(InitialStrategy):
    """An initial strategy that selects the candidates at random."""

    TYPE = "RANDOM"

    def recommend(self, candidates: pd.DataFrame, batch_quantity: int = 1) -> pd.Index:
        """Uniform random selection of candidates."""
        return pd.Index(
            np.random.choice(candidates.index, batch_quantity, replace=False)
        )


class StrategyConfig(BaseModel, extra=Extra.forbid, arbitrary_types_allowed=True):
    """Configuration class for creating strategy objects."""

    # TODO: consider adding validators for the individual component classes of the
    #  strategy or introducing config classes for them (-> disable arbitrary types)

    surrogate_model: Union[str, Type[SurrogateModel]] = "GP"
    acquisition_function: Union[Literal["EI"], Type[AcquisitionFunction]] = "EI"
    initial_strategy: Union[str, InitialStrategy] = "RANDOM"
    recommender: Union[str, Type[Recommender]] = "RANKING"

    @validator("surrogate_model", always=True)
    def validate_type(cls, model):
        """Validates if the given surrogate model type exists."""
        if isinstance(model, str):
            check_if_in(model, SurrogateModel.SUBCLASSES)
            return SurrogateModel.SUBCLASSES[model]
        return model

    @validator("acquisition_function", always=True)
    def validate_acquisition_function(cls, fun):
        """Validates if the given acquisition function type exists."""
        # TODO: change once an acquisition wrapper class has been introduced
        if isinstance(fun, str) and fun == "EI":
            return ExpectedImprovement
        return fun

    @validator("initial_strategy", always=True)
    def validate_initial_strategy(cls, strategy):
        """Validates if the given initial strategy type exists."""
        if isinstance(strategy, str):
            check_if_in(strategy, InitialStrategy.SUBCLASSES)
            return InitialStrategy.SUBCLASSES[strategy]()
        return strategy

    @validator("recommender", always=True)
    def validate_recommender(cls, recommender):
        """Validates if the given recommender model type exists."""
        if isinstance(recommender, str):
            check_if_in(recommender, Recommender.SUBCLASSES)
            return Recommender.SUBCLASSES[recommender]
        return recommender


class Strategy:
    """Abstract base class for all DoE strategies."""

    def __init__(self, config: StrategyConfig):
        # process input arguments
        self.surrogate_model = config.surrogate_model
        self.acquisition_function = config.acquisition_function
        self.initial_strategy = config.initial_strategy
        self.recommender = config.recommender

        # declare remaining members
        self._surrogate_model: Optional[SurrogateModel] = None
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
            self._surrogate_model = self.surrogate_model()
            self._surrogate_model.fit(train_x, train_y)
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
        acqf = self.acquisition_function(self._surrogate_model.model, self._best_f)

        # select the next experiments using the given recommender approach
        recommender = self.recommender(acqf)
        idxs = recommender.recommend(candidates, batch_quantity)

        return idxs
