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

from .utils import isabstract, to_tensor


class NoMCAcquisitionFunctionError(Exception):
    """An exception raised when a Monte Carlo acquisition function is required
    but an analytical acquisition function has been selected by the user."""


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
    def recommend(self, candidates: pd.DataFrame, batch_quantity: int = 1) -> pd.Index:
        """
        Recommends the next experiments to be conducted.

        Parameters
        ----------
        candidates : pd.DataFrame
            The features of all candidate experiments that could be conducted next.
        batch_quantity : int
            The number of experiments to be conducted in parallel.

        Returns
        -------
        The DataFrame indices of the specific experiments selected.
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

    def recommend(self, candidates: pd.DataFrame, batch_quantity: int = 1) -> pd.Index:
        """See base class."""
        # determine the next set of points to be tested
        candidates_tensor = to_tensor(candidates)
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
        locs = pd.Index(
            pd.merge(
                candidates.reset_index(),
                pd.DataFrame(points, columns=candidates.columns),
                on=list(candidates),
            )["index"]
        )

        return locs


class MarginalRankingRecommender(Recommender):
    """
    Recommends the top experiments from the ranking obtained by evaluating the
    acquisition function on the marginal posterior predictive distribution of each
    candidate, i.e. by computing the score for each candidate individually in a
    non-batch fashion.
    """

    type = "UNRESTRICTED_RANKING"

    def recommend(self, candidates: pd.DataFrame, batch_quantity: int = 1) -> pd.Index:
        """See base class."""
        # prepare the candidates in t-batches (= parallel marginal evaluation)
        candidates_tensor = to_tensor(candidates).unsqueeze(1)

        # evaluate the acquisition function for each t-batch and construct the ranking
        acqf_values = self.acquisition_function(candidates_tensor)
        ilocs = torch.argsort(acqf_values, descending=True)

        # return the dataframe indices of the top ranked candidates
        locs = candidates.index[ilocs[:batch_quantity].numpy()]
        return locs


class RandomRecommender(Recommender):
    """
    Recommends experiments randomly.
    """

    type = "RANDOM"

    def recommend(self, candidates: pd.DataFrame, batch_quantity: int = 1) -> pd.Index:
        """See base class."""

        return candidates.sample(n=batch_quantity).index
