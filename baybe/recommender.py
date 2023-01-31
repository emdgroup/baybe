# pylint: disable=too-few-public-methods
"""
Recommender classes for optimizing acquisition functions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Type

import pandas as pd
import torch
from botorch.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf_discrete

from .searchspace import SearchSpace
from .utils import isabstract, NoMCAcquisitionFunctionError, to_tensor


class Recommender(ABC):
    """
    Abstract base class for all recommenders.

    The job of a recommender is to select (i.e. "recommend") a subset of candidate
    experiments based on an underlying (batch) acquisition criterion.
    """

    type: str
    SUBCLASSES: Dict[str, Type[Recommender]] = {}

    def __init__(self, acquisition_function: Optional[AcquisitionFunction]):
        self.acquisition_function = acquisition_function

    @classmethod
    def __init_subclass__(cls, **kwargs):
        """Registers new subclasses dynamically."""
        super().__init_subclass__(**kwargs)
        if not isabstract(cls):
            cls.SUBCLASSES[cls.type] = cls

    @abstractmethod
    def recommend(
        self,
        searchspace: SearchSpace,
        batch_quantity: int = 1,
        allow_repeated_recommendations: bool = False,
        allow_recommending_already_measured: bool = True,
    ) -> pd.DataFrame:
        """
        Recommends the next experiments to be conducted.

        Parameters
        ----------
        searchspace : SearchSpace
            The search space from which recommendations should be provided.
        batch_quantity : int
            The number of experiments to be conducted in parallel.

        Returns
        -------
        The DataFrame with the specific experiments recommended.
        """


class SequentialGreedyRecommender(Recommender):
    """
    Recommends the next set of experiments by means of sequential greedy optimization,
    i.e. using a growing set of candidates points, where each new candidate is
    selected by optimizing the joint acquisition score of the current candidate set
    while having fixed all previous candidates.

    Note: This approach only works with Monte Carlo acquisition functions, i.e. of
        type `botorch.acquisition.monte_carlo.MCAcquisitionFunction`.
    """

    type = "SEQUENTIAL_GREEDY"

    def recommend(
        self,
        searchspace: SearchSpace,
        batch_quantity: int = 1,
        allow_repeated_recommendations: bool = False,
        allow_recommending_already_measured: bool = True,
    ) -> pd.DataFrame:
        """See base class."""
        candidates_exp, candidates_comp = searchspace.discrete.get_candidates(
            allow_repeated_recommendations,
            allow_recommending_already_measured,
        )

        # determine the next set of points to be tested
        candidates_tensor = to_tensor(candidates_comp)
        try:
            points, _ = optimize_acqf_discrete(
                self.acquisition_function, batch_quantity, candidates_tensor
            )
        except AttributeError as ex:
            raise NoMCAcquisitionFunctionError(
                f"The '{self.__class__.__name__}' only works with Monte Carlo "
                f"acquisition functions."
            ) from ex

        # retrieve the index of the points from the input dataframe
        # TODO: This additional inelegant step is unfortunately required since BoTorch
        #   does not return the indices of the points. However, as soon as we move to
        #   continuous spaces, we will have to use another representation anyway
        #   (which is also the reason why BoTorch does not support it).
        # IMPROVE: The merging procedure is conceptually similar to what
        #   `SearchSpace._match_measurement_with_searchspace_indices` does, though using
        #   a simpler matching logic. When refactoring the SearchSpace class to
        #   handle continuous parameters, a corresponding utility could be extracted.
        idxs = pd.Index(
            pd.merge(
                candidates_comp.reset_index(),
                pd.DataFrame(points, columns=candidates_comp.columns),
                on=list(candidates_comp),
            )["index"]
        )
        assert len(points) == len(idxs)

        rec = candidates_exp.loc[idxs, :]
        searchspace.discrete.metadata.loc[idxs, "was_recommended"] = True

        return rec


class MarginalRankingRecommender(Recommender):
    """
    Recommends the top experiments from the ranking obtained by evaluating the
    acquisition function on the marginal posterior predictive distribution of each
    candidate, i.e. by computing the score for each candidate individually in a
    non-batch fashion.
    """

    type = "UNRESTRICTED_RANKING"

    def recommend(
        self,
        searchspace: SearchSpace,
        batch_quantity: int = 1,
        allow_repeated_recommendations: bool = False,
        allow_recommending_already_measured: bool = True,
    ) -> pd.DataFrame:
        """See base class."""
        candidates_exp, candidates_comp = searchspace.discrete.get_candidates(
            allow_repeated_recommendations,
            allow_recommending_already_measured,
        )

        # prepare the candidates in t-batches (= parallel marginal evaluation)
        candidates_tensor = to_tensor(candidates_comp).unsqueeze(1)

        # evaluate the acquisition function for each t-batch and construct the ranking
        acqf_values = self.acquisition_function(candidates_tensor)
        ilocs = torch.argsort(acqf_values, descending=True)

        # return top ranked candidates
        idxs = candidates_comp.index[ilocs[:batch_quantity].numpy()]
        rec = candidates_exp.loc[idxs, :]
        searchspace.discrete.metadata.loc[idxs, "was_recommended"] = True

        return rec


class RandomRecommender(Recommender):
    """
    Recommends experiments randomly.
    """

    type = "RANDOM"

    def recommend(
        self,
        searchspace: SearchSpace,
        batch_quantity: int = 1,
        allow_repeated_recommendations: bool = False,
        allow_recommending_already_measured: bool = True,
    ) -> pd.DataFrame:
        """See base class."""
        candidates_exp, _ = searchspace.discrete.get_candidates(
            allow_repeated_recommendations,
            allow_recommending_already_measured,
        )

        # randomly select from discrete candidates
        idxs = candidates_exp.sample(n=batch_quantity).index
        rec = candidates_exp.loc[idxs, :]
        searchspace.discrete.metadata.loc[idxs, "was_recommended"] = True

        return rec
