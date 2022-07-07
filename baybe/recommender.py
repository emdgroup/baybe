# pylint: disable=too-few-public-methods
"""
Recommender classes for optimizing acquisition functions.
"""

from abc import ABC, abstractmethod

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

    def __init__(
        self,
        acquisition_function: AcquisitionFunction,
    ):
        self.acquisition_function = acquisition_function

    @abstractmethod
    def recommend(self, candidates: pd.DataFrame, batch_size: int = 1) -> pd.Index:
        """
        Recommends the next experiments to be conducted.

        Parameters
        ----------
        candidates : pd.DataFrame
            The features of all candidate experiments that could be conducted next.
        batch_size : int
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

    def recommend(self, candidates: pd.DataFrame, batch_size: int = 1) -> pd.Index:
        """See base class."""
        # prepare the candidates in t-batches
        candidates_tensor = to_tensor(candidates).unsqueeze(1)

        # evaluate the acquisition function for each t-batch and construct the ranking
        acqf_values = self.acquisition_function(candidates_tensor)
        ilocs = torch.argsort(acqf_values)

        # return the dataframe indices of the top ranked candidates
        locs = candidates.index[ilocs[:batch_size].numpy()]
        return locs
