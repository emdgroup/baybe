# pylint: disable=not-callable, no-member  # TODO: due to validators --> find fix
"""
Strategies for Design of Experiments (DOE).
"""

from __future__ import annotations

from functools import partial
from typing import Literal

import pandas as pd
from botorch.acquisition import (
    ExpectedImprovement,
    PosteriorMean,
    ProbabilityOfImprovement,
    qExpectedImprovement,
    qProbabilityOfImprovement,
    qUpperConfidenceBound,
    UpperConfidenceBound,
)
from pydantic import Extra, validator

from .acquisition import debotorchize
from .recommender import Recommender
from .searchspace import SearchSpace
from .surrogate import SurrogateModel
from .utils import BaseModel, check_if_in, to_tensor


class Strategy(BaseModel, extra=Extra.forbid, arbitrary_types_allowed=True):
    """Abstract base class for all DOE strategies."""

    # TODO: consider adding validators for the individual component classes of the
    #  strategy or introducing config classes for them (-> disable arbitrary types)

    # object variables
    searchspace: SearchSpace
    surrogate_model_cls: str = "GP"
    acquisition_function_cls: Literal[
        "PM", "PI", "EI", "UCB", "qPI", "qEI", "qUCB"
    ] = "qEI"  # TODO: automatic selection between EI and qEI depending on query size
    initial_recommender_cls: str = "RANDOM"
    recommender_cls: str = "SEQUENTIAL_GREEDY_DISCRETE"
    allow_repeated_recommendations: bool = True
    allow_recommending_already_measured: bool = True
    numerical_measurements_must_be_within_tolerance: bool = True

    # TODO: introduce a reusable validator once they all perform the same operation

    @validator("surrogate_model_cls", always=True)
    def validate_surrogate_model(cls, model):
        """Validates if the given surrogate model type exists."""
        check_if_in(model, list(SurrogateModel.SUBCLASSES.keys()))
        return model

    def get_surrogate_model_cls(self):  # pylint: disable=missing-function-docstring
        # TODO: work in progress
        return SurrogateModel.SUBCLASSES[self.surrogate_model_cls]

    def get_acqusition_function_cls(self):  # pylint: disable=missing-function-docstring
        # TODO: work in progress
        mapping = {
            "PM": PosteriorMean,
            "PI": ProbabilityOfImprovement,
            "EI": ExpectedImprovement,
            "UCB": partial(UpperConfidenceBound, beta=1.0),
            "qEI": qExpectedImprovement,
            "qPI": qProbabilityOfImprovement,
            "qUCB": partial(qUpperConfidenceBound, beta=1.0),
        }
        fun = debotorchize(mapping[self.acquisition_function_cls])
        return fun

    @validator("initial_recommender_cls", always=True)
    def validate_initial_recommender_cls(cls, initial_recommender_cls):
        """Validates if the given initial recommender type exists."""
        check_if_in(
            initial_recommender_cls,
            [
                key
                for key, subclass in Recommender.SUBCLASSES.items()
                if subclass.is_model_free
            ],
        )
        return initial_recommender_cls

    def get_initial_recommender_cls(self):  # pylint: disable=missing-function-docstring
        # TODO: work in progress
        return Recommender.SUBCLASSES[self.initial_recommender_cls]

    @validator("recommender_cls", always=True)
    def validate_recommender(cls, recommender):
        """Validates if the given recommender model type exists."""
        check_if_in(recommender, list(Recommender.SUBCLASSES.keys()))
        return recommender

    def get_recommender_cls(self):  # pylint: disable=missing-function-docstring
        # TODO: work in progress
        return Recommender.SUBCLASSES[self.recommender_cls]

    def _fit(self, train_x: pd.DataFrame, train_y: pd.DataFrame) -> SurrogateModel:
        """
        Uses the given data to train a fresh surrogate model instance for the DOE
        strategy.

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

        surrogate_model_cls = self.get_surrogate_model_cls()
        surrogate_model = surrogate_model_cls(self.searchspace)
        surrogate_model.fit(*to_tensor(train_x, train_y))

        return surrogate_model

    def recommend(
        self,
        train_x: pd.DataFrame,
        train_y: pd.DataFrame,
        batch_quantity: int = 1,
    ) -> pd.DataFrame:
        """
        Recommends the next experiments to be conducted.

        Parameters
        ----------
        train_x : pd.DataFrame
            The features of the conducted experiments.
        train_y : pd.DataFrame
            The corresponding response values.
        batch_quantity : int (default = 1)
            The number of experiments to be conducted in parallel.

        Returns
        -------
        The DataFrame with the specific experiments recommended.
        """
        # Special treatment of initial recommendation
        if len(train_x) == 0:
            initial_recommender_cls = self.get_initial_recommender_cls()
            recommender = initial_recommender_cls(
                searchspace=self.searchspace, acquisition_function=None
            )
        else:
            # construct the acquisition function
            if self.recommender_cls == "RANDOM":
                acqf = None
            else:
                best_f = train_y.max()
                surrogate_model = self._fit(train_x, train_y)
                acquisition_function_cls = self.get_acqusition_function_cls()
                acqf = acquisition_function_cls(surrogate_model, best_f)

            # select the next experiments using the given recommender approach
            recommender_cls = self.get_recommender_cls()
            recommender = recommender_cls(
                searchspace=self.searchspace, acquisition_function=acqf
            )

        rec = recommender.recommend(
            batch_quantity,
            self.allow_repeated_recommendations,
            self.allow_recommending_already_measured,
        )

        return rec
