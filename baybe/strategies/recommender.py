# pylint: disable=missing-class-docstring, missing-function-docstring
# TODO: add docstrings

"""Base classes for all recommenders."""

from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, ClassVar, Literal, Optional

import cattrs
import pandas as pd

from attrs import define
from botorch.acquisition import (
    ExpectedImprovement,
    PosteriorMean,
    ProbabilityOfImprovement,
    qExpectedImprovement,
    qProbabilityOfImprovement,
    qUpperConfidenceBound,
    UpperConfidenceBound,
)

from baybe.acquisition import debotorchize
from baybe.searchspace import SearchSpace, SearchSpaceType
from baybe.surrogate import SurrogateModel
from baybe.utils import (
    get_base_unstructure_hook,
    NotEnoughPointsLeftError,
    to_tensor,
    unstructure_base,
)


# TODO: See if the there is a more elegant way to share this functionality
#   among all purely discrete recommenders (without introducing complicates class
#   hierarchies).
def select_candidates_and_recommend(
    searchspace: SearchSpace,
    recommend: Callable,
    batch_quantity: int = 1,
    allow_repeated_recommendations: bool = False,
    allow_recommending_already_measured: bool = True,
) -> pd.DataFrame:
    # Get discrete candidates. The metadata flags are ignored if the searchspace
    # has a continuous component.
    _, candidates_comp = searchspace.discrete.get_candidates(
        allow_repeated_recommendations=allow_repeated_recommendations
        or not searchspace.continuous.empty,
        allow_recommending_already_measured=allow_recommending_already_measured
        or not searchspace.continuous.empty,
    )

    # Check if enough candidates are left
    if len(candidates_comp) < batch_quantity:
        raise NotEnoughPointsLeftError(
            f"Using the current settings, there are fewer than {batch_quantity} "
            "possible data points left to recommend. This can be "
            "either because all data points have been measured at some point "
            "(while 'allow_repeated_recommendations' or "
            "'allow_recommending_already_measured' being False) "
            "or because all data points are marked as 'dont_recommend'."
        )

    # Get recommendations
    idxs = recommend(searchspace, candidates_comp, batch_quantity)
    rec = searchspace.discrete.exp_rep.loc[idxs, :]

    # Update metadata
    searchspace.discrete.metadata.loc[idxs, "was_recommended"] = True

    # Return recommendations
    return rec


@define
class Recommender(ABC):

    compatibility: ClassVar[SearchSpaceType]

    @abstractmethod
    def recommend(
        self,
        searchspace: SearchSpace,
        batch_quantity: int = 1,
        train_x: Optional[pd.DataFrame] = None,
        train_y: Optional[pd.DataFrame] = None,
        allow_repeated_recommendations: bool = False,
        allow_recommending_already_measured: bool = True,
    ) -> pd.DataFrame:
        pass


@define
class NonPredictiveRecommender(Recommender, ABC):
    def recommend(
        self,
        searchspace: SearchSpace,
        batch_quantity: int = 1,
        train_x: Optional[pd.DataFrame] = None,
        train_y: Optional[pd.DataFrame] = None,
        allow_repeated_recommendations: bool = False,
        allow_recommending_already_measured: bool = True,
    ) -> pd.DataFrame:

        if searchspace.type == SearchSpaceType.DISCRETE:
            return select_candidates_and_recommend(
                searchspace,
                self._recommend_discrete,
                batch_quantity,
                allow_repeated_recommendations,
                allow_recommending_already_measured,
            )
        if searchspace.type == SearchSpaceType.CONTINUOUS:
            return self._recommend_continuous(searchspace, batch_quantity)
        raise NotImplementedError()

    def _recommend_discrete(
        self,
        searchspace: SearchSpace,
        candidates_comp: pd.DataFrame,
        batch_quantity: int,
    ):
        raise NotImplementedError()

    def _recommend_continuous(self, searchspace: SearchSpace, batch_quantity: int):
        raise NotImplementedError()

    def _recommend_hybrid(self, searchspace: SearchSpace, batch_quantity: int):
        raise NotImplementedError()


@define
class BayesianRecommender(Recommender, ABC):

    surrogate_model_cls: str = "GP"
    acquisition_function_cls: Literal[
        "PM", "PI", "EI", "UCB", "qPI", "qEI", "qUCB"
    ] = "qEI"

    def get_acquisition_function_cls(
        self,
    ):  # pylint: disable=missing-function-docstring
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

    def _fit(
        self,
        searchspace: SearchSpace,
        train_x: pd.DataFrame,
        train_y: pd.DataFrame,
    ) -> SurrogateModel:
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
        surrogate_model = surrogate_model_cls(searchspace)
        surrogate_model.fit(*to_tensor(train_x, train_y))

        return surrogate_model

    def get_surrogate_model_cls(self):  # pylint: disable=missing-function-docstring
        # TODO: work in progress
        return SurrogateModel.SUBCLASSES[self.surrogate_model_cls]

    def recommend(
        self,
        searchspace: SearchSpace,
        batch_quantity: int = 1,
        train_x: Optional[pd.DataFrame] = None,
        train_y: Optional[pd.DataFrame] = None,
        allow_repeated_recommendations: bool = False,
        allow_recommending_already_measured: bool = True,
    ) -> pd.DataFrame:

        best_f = train_y.max()
        surrogate_model = self._fit(searchspace, train_x, train_y)
        acquisition_function_cls = self.get_acquisition_function_cls()
        acqf = acquisition_function_cls(surrogate_model, best_f)

        if searchspace.type == SearchSpaceType.DISCRETE:
            return select_candidates_and_recommend(
                searchspace,
                partial(self._recommend_discrete, acqf),
                batch_quantity,
                allow_repeated_recommendations,
                allow_recommending_already_measured,
            )
        if searchspace.type == SearchSpaceType.CONTINUOUS:
            return self._recommend_continuous(acqf, searchspace, batch_quantity)
        raise NotImplementedError()

    def _recommend_discrete(
        self,
        acquisition_function: Callable,
        searchspace: SearchSpace,
        candidates_comp: pd.DataFrame,
        batch_quantity: int,
    ):
        raise NotImplementedError()

    def _recommend_continuous(
        self,
        acquisition_function: Callable,
        searchspace: SearchSpace,
        batch_quantity: int,
    ):
        raise NotImplementedError()

    def _recommend_hybrid(
        self,
        acquisition_function: Callable,
        searchspace: SearchSpace,
        batch_quantity: int,
    ):
        raise NotImplementedError()


# Register (un-)structure hooks
cattrs.register_unstructure_hook(Recommender, unstructure_base)
cattrs.register_structure_hook(Recommender, get_base_unstructure_hook(Recommender))
