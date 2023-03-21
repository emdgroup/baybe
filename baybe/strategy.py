# pylint: disable=not-callable, no-member  # TODO: due to validators --> find fix
"""
Strategies for Design of Experiments (DOE).
"""

from __future__ import annotations

from functools import partial
from typing import Literal, Optional, Type, Union

import pandas as pd
from botorch.acquisition import (
    AcquisitionFunction,
    ExpectedImprovement,
    PosteriorMean,
    ProbabilityOfImprovement,
    qExpectedImprovement,
    qProbabilityOfImprovement,
    qUpperConfidenceBound,
    UpperConfidenceBound,
)
from pydantic import BaseModel, Extra, validator

from .acquisition import debotorchize
from .recommender import ModelFreeRecommender, Recommender
from .searchspace import SearchSpace
from .surrogate import SurrogateModel
from .utils import check_if_in, to_tensor


class Strategy(BaseModel, extra=Extra.forbid, arbitrary_types_allowed=True):
    """Abstract base class for all DOE strategies."""

    # TODO: consider adding validators for the individual component classes of the
    #  strategy or introducing config classes for them (-> disable arbitrary types)

    # object variables
    searchspace: SearchSpace
    surrogate_model_cls: Union[str, Type[SurrogateModel]] = "GP"
    acquisition_function_cls: Union[
        Literal["PM", "PI", "EI", "UCB", "qPI", "qEI", "qUCB"],
        Type[AcquisitionFunction],
    ] = "qEI"  # TODO: automatic selection between EI and qEI depending on query size
    initial_recommender_cls: Union[str, Type[ModelFreeRecommender]] = "RANDOM"
    recommender_cls: Union[str, Type[Recommender]] = "SEQUENTIAL_GREEDY"

    # TODO: The following member declarations become obsolete in pydantic 2.0 when
    #  __post_init_post_parse__ is available:
    #   - https://github.com/samuelcolvin/pydantic/issues/691
    #   - https://github.com/samuelcolvin/pydantic/issues/1729
    surrogate_model: Optional[SurrogateModel] = None
    best_f: Optional[float] = None
    use_initial_recommender: bool = True

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
            # TODO: make beta a configurable parameter
            mapping = {
                "PM": PosteriorMean,
                "PI": ProbabilityOfImprovement,
                "EI": ExpectedImprovement,
                "UCB": partial(UpperConfidenceBound, beta=1.0),
                "qEI": qExpectedImprovement,
                "qPI": qProbabilityOfImprovement,
                "qUCB": partial(qUpperConfidenceBound, beta=1.0),
            }
            fun = debotorchize(mapping[fun])
        return fun

    @validator("initial_recommender_cls", always=True)
    def validate_initial_recommender_cls(cls, initial_recommender_cls):
        """Validates if the given initial recommender type exists."""
        if isinstance(initial_recommender_cls, str):
            check_if_in(
                initial_recommender_cls,
                [
                    key
                    for key, subclass in Recommender.SUBCLASSES.items()
                    if subclass.is_model_free
                ],
            )
            return Recommender.SUBCLASSES[initial_recommender_cls]
        return initial_recommender_cls

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
        self.use_initial_recommender = len(train_x) == 0

        # if data is provided (and the strategy is not random), train the surrogate
        if (not self.use_initial_recommender) and (
            self.recommender_cls.type != "RANDOM"
        ):
            self.surrogate_model = self.surrogate_model_cls(self.searchspace)
            self.surrogate_model.fit(*to_tensor(train_x, train_y))
            self.best_f = train_y.max()

    def recommend(
        self,
        batch_quantity: int = 1,
        allow_repeated_recommendations: bool = False,
        allow_recommending_already_measured: bool = True,
    ) -> pd.DataFrame:
        """
        Recommends the next experiments to be conducted.

        Parameters
        ----------
        batch_quantity : int (default = 1)
            The number of experiments to be conducted in parallel.
        allow_repeated_recommendations : bool
            Whether points whose discrete parts were already recommended can be
            recommended again.
        allow_recommending_already_measured : bool
            Whether points whose discrete parts were already measured can be
            recommended again.

        Returns
        -------
        The DataFrame with the specific experiments recommended.
        """
        # Special treatment of initial recommendation
        if self.use_initial_recommender:
            recommender = self.initial_recommender_cls(
                searchspace=self.searchspace, acquisition_function=None
            )
        else:
            # construct the acquisition function
            acqf = (
                None
                if self.recommender_cls.type == "RANDOM"
                else self.acquisition_function_cls(self.surrogate_model, self.best_f)
            )

            # select the next experiments using the given recommender approach
            recommender = self.recommender_cls(
                searchspace=self.searchspace, acquisition_function=acqf
            )

        rec = recommender.recommend(
            batch_quantity,
            allow_repeated_recommendations,
            allow_recommending_already_measured,
        )

        return rec
