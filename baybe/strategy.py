# pylint: disable=not-callable, no-member  # TODO: due to validators --> find fix
"""
Strategies for Design of Experiments (DOE).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial
from typing import Dict, Literal, Optional, Type, Union

import numpy as np
import pandas as pd
from botorch.acquisition import (
    AcquisitionFunction,
    ExpectedImprovement,
    PosteriorMean,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from pydantic import BaseModel, Extra, validator

from .acquisition import debotorchize
from .recommender import Recommender
from .surrogate import SurrogateModel
from .utils import check_if_in, to_tensor
from .utils.sampling_algorithms import _dpp, _fps


class InitialStrategy(ABC):
    """
    Abstract base class for all initial design strategies. They are used for selecting
    initial sets of candidate experiments, i.e. without considering experimental data.
    """

    type: str
    SUBCLASSES: Dict[str, Type[InitialStrategy]] = {}

    @abstractmethod
    def recommend(self, candidates: pd.DataFrame, batch_quantity: int = 1) -> pd.Index:
        """
        Selects a first subset of points from the given candidates.

        Parameters
        ----------
        candidates : pd.DataFrame
            The features of all candidate experiments that could be conducted.
        batch_quantity : int (default = 1)
            The number of experiments to be conducted in parallel.

        Returns
        -------
        The DataFrame indices of the specific experiments selected.
        """

    @classmethod
    def __init_subclass__(cls, **kwargs):
        """Registers new subclasses dynamically."""
        super().__init_subclass__(**kwargs)
        cls.SUBCLASSES[cls.type] = cls


class RandomInitialStrategy(InitialStrategy):
    """An initial strategy that selects candidates uniformly at random."""

    type = "RANDOM"

    def recommend(self, candidates: pd.DataFrame, batch_quantity: int = 1) -> pd.Index:
        """Uniform random selection of candidates."""
        return pd.Index(
            np.random.choice(candidates.index, batch_quantity, replace=False)
        )


class DPPInitialStrategy(InitialStrategy):
    """An initial strategy that selects the candidates using determinantal point process
    algorithm."""

    type = "DPP"

    def recommend(self, candidates: pd.DataFrame, batch_quantity: int = 1) -> pd.Index:
        """Uniform random selection of candidates."""
        return _dpp(
            candidates, batch_quantity, kernel="RBF", epsilon=1e-10, start_index=1
        )


class FPSInitialStrategy(InitialStrategy):
    """An initial strategy that selects the candidates using farthest point sampling
    algorithm."""

    type = "FPS"

    def recommend(self, candidates: pd.DataFrame, batch_quantity: int = 1) -> pd.Index:
        """Uniform random selection of candidates."""
        return _fps(candidates, batch_quantity, start_index=1)


class Strategy(BaseModel, extra=Extra.forbid, arbitrary_types_allowed=True):
    """Abstract base class for all DOE strategies."""

    # TODO: consider adding validators for the individual component classes of the
    #  strategy or introducing config classes for them (-> disable arbitrary types)

    # object variables
    searchspace: pd.DataFrame
    surrogate_model_cls: Union[str, Type[SurrogateModel]] = "GP"
    acquisition_function_cls: Union[
        Literal["PM", "PI", "EI", "UCB"], Type[AcquisitionFunction]
    ] = "EI"
    initial_strategy: Union[str, InitialStrategy] = "RANDOM"
    recommender_cls: Union[str, Type[Recommender]] = "UNRESTRICTED_RANKING"

    # TODO: The following member declarations become obsolete in pydantic 2.0 when
    #  __post_init_post_parse__ is available:
    #   - https://github.com/samuelcolvin/pydantic/issues/691
    #   - https://github.com/samuelcolvin/pydantic/issues/1729
    surrogate_model: Optional[SurrogateModel] = None
    best_f: Optional[float] = None
    use_initial_strategy: bool = True

    # TODO: introduce a reusable validator once they all perform the same operation

    @validator("surrogate_model_cls", always=True)
    def validate_surrogate_model(cls, model):
        """Validates if the given surrogate model type exists."""
        if isinstance(model, str):
            check_if_in(model, list(SurrogateModel.SUBCLASSES.keys()))
            return SurrogateModel.SUBCLASSES[model]
        return model

    @validator("acquisition_function_cls", always=True)
    def validate_acquisition_function(cls, fun):
        """Validates if the given acquisition function type exists."""
        if isinstance(fun, str):
            mapping = {
                "PM": PosteriorMean,
                "PI": ProbabilityOfImprovement,
                "EI": ExpectedImprovement,
                "UCB": partial(UpperConfidenceBound, beta=1.0),
            }
            fun = debotorchize(mapping[fun])
        return fun

    @validator("initial_strategy", always=True)
    def validate_initial_strategy(cls, init_strategy):
        """Validates if the given initial strategy type exists."""
        if isinstance(init_strategy, str):
            check_if_in(init_strategy, list(InitialStrategy.SUBCLASSES.keys()))
            return InitialStrategy.SUBCLASSES[init_strategy]()
        return init_strategy

    @validator("recommender_cls", always=True)
    def validate_recommender(cls, recommender):
        """Validates if the given recommender model type exists."""
        if isinstance(recommender, str):
            check_if_in(recommender, list(Recommender.SUBCLASSES.keys()))
            return Recommender.SUBCLASSES[recommender]
        return recommender

    def fit(self, train_x: pd.DataFrame, train_y: pd.DataFrame) -> None:
        """
        Uses the given data to train a fresh surrogate model instance for the DOE
        strategy. If available, previous training data will be overwritten.

        Parameters
        ----------
        train_x : pd.DataFrame
            The features of the conducted experiments.
        train_y : pd.DataFrame
            The corresponding response values.
        """
        # validate input
        if not train_x.index.equals(train_y.index):
            raise ValueError("Training inputs and targets must have the same index.")

        # if no data is provided, apply the initial selection strategy
        self.use_initial_strategy = len(train_x) == 0

        # if data is provided (and the strategy is not random), train the surrogate
        if (not self.use_initial_strategy) and (self.recommender_cls.type != "RANDOM"):
            self.surrogate_model = self.surrogate_model_cls(self.searchspace)
            self.surrogate_model.fit(*to_tensor(train_x, train_y))
            self.best_f = train_y.max()

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
        The DataFrame indices of the specific experiments selected.
        """
        # if no training data exists, apply the strategy for initial recommendations
        if self.use_initial_strategy:
            return self.initial_strategy.recommend(candidates, batch_quantity)

        # construct the acquisition function
        acqf = (
            self.acquisition_function_cls(self.surrogate_model, self.best_f)
            if self.recommender_cls.type != "RANDOM"
            else None
        )

        # select the next experiments using the given recommender approach
        recommender = self.recommender_cls(acqf)
        idxs = recommender.recommend(candidates, batch_quantity)

        return idxs
