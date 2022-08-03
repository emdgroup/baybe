# pylint: disable=too-few-public-methods
"""
Recommender classes for optimizing acquisition functions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import pandas as pd
import torch
from botorch.acquisition import AcquisitionFunction

from baybe.utils import to_tensor

# TODO: use botorch's built-in acquisition optimization methods
#   (problem: they do not return the indices but the candidate points)


class Recommender(ABC):
    """
    Abstract base class for all Recommenders.

    The job of a Recommender is to select (i.e. "recommend") a subset of candidate
    experiments based on an underlying (batch) acquisition criterion.
    """

    type: str
    SUBCLASSES: Dict[str, Recommender] = {}

    def __init__(self, acquisition_function: AcquisitionFunction):
        self.acquisition_function = acquisition_function

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
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


class MarginalRankingRecommender(Recommender):
    """
    Recommends the top experiments from the ranking obtained by evaluating the
    acquisition function on the marginal posterior distribution of each candidate,
    i.e. by computing the score for each candidate individually in a non-batch
    fashion.
    """

    type = "UNRESTRICTED_RANKING"

    def recommend(self, candidates: pd.DataFrame, batch_quantity: int = 1) -> pd.Index:
        """See base class."""
        # prepare the candidates in t-batches
        candidates_tensor = to_tensor(candidates).unsqueeze(1)

        # evaluate the acquisition function for each t-batch and construct the ranking
        acqf_values = self.acquisition_function(candidates_tensor)
        ilocs = torch.argsort(acqf_values)

        # return the dataframe indices of the top ranked candidates
        locs = candidates.index[ilocs[:batch_quantity].numpy()]
        return locs
