# pylint: disable=not-callable, no-member  # TODO: due to validators --> find fix
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

    type: str
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
        cls.SUBCLASSES[cls.type] = cls


class RandomInitialStrategy(InitialStrategy):
    """An initial strategy that selects the candidates at random."""

    type = "RANDOM"

    def recommend(self, candidates: pd.DataFrame, batch_quantity: int = 1) -> pd.Index:
        """Uniform random selection of candidates."""
        return pd.Index(
            np.random.choice(candidates.index, batch_quantity, replace=False)
        )


class Strategy(BaseModel, extra=Extra.forbid, arbitrary_types_allowed=True):
    """Abstract base class for all DoE strategies."""

    # TODO: consider adding validators for the individual component classes of the
    #  strategy or introducing config classes for them (-> disable arbitrary types)

    # object variables
    surrogate_model_cls: Union[str, Type[SurrogateModel]] = "GP"
    acquisition_function_cls: Union[Literal["EI"], Type[AcquisitionFunction]] = "EI"
    initial_strategy: Union[str, InitialStrategy] = "RANDOM"
    recommender_cls: Union[str, Type[Recommender]] = "RANKING"

    # TODO: this becomes obsolete in pydantic 2.0 when the __post_init_post_parse__
    #   is available:
    #   - https://github.com/samuelcolvin/pydantic/issues/691
    #   - https://github.com/samuelcolvin/pydantic/issues/1729
    surrogate_model: Optional[SurrogateModel] = None
    best_f: Optional[float] = None
    use_initial_strategy: bool = True

    @validator("surrogate_model_cls", always=True)
    def validate_type(cls, model):
        """Validates if the given surrogate model type exists."""
        if isinstance(model, str):
            check_if_in(model, list(SurrogateModel.SUBCLASSES.keys()))
            return SurrogateModel.SUBCLASSES[model]
        return model

    @validator("acquisition_function_cls", always=True)
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
            check_if_in(strategy, list(InitialStrategy.SUBCLASSES.keys()))
            return InitialStrategy.SUBCLASSES[strategy]()
        return strategy

    @validator("recommender_cls", always=True)
    def validate_recommender(cls, recommender):
        """Validates if the given recommender model type exists."""
        if isinstance(recommender, str):
            check_if_in(recommender, list(Recommender.SUBCLASSES.keys()))
            return Recommender.SUBCLASSES[recommender]
        return recommender

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
        self.use_initial_strategy = len(train_x) == 0
        if not self.use_initial_strategy:
            self.surrogate_model = self.surrogate_model_cls()
            self.surrogate_model.fit(train_x, train_y)
            self.best_f = train_y.min()

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

        # if no training data exists, apply the strategy for initial recommendations
        if self.use_initial_strategy:
            return self.initial_strategy.recommend(candidates, batch_quantity)

        # construct the acquisition function
        # TODO: the current approach only works for gpytorch GP surrogate models
        #   (for other surrogate models, some wrapper is required)
        acqf = self.acquisition_function_cls(self.surrogate_model.model, self.best_f)

        # select the next experiments using the given recommender approach
        recommender = self.recommender_cls(acqf)
        idxs = recommender.recommend(candidates, batch_quantity)

        return idxs
