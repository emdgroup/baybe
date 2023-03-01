# pylint: disable=too-few-public-methods
"""
Recommender classes for optimizing acquisition functions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Dict, Optional, Type

import pandas as pd
import torch
from botorch.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf, optimize_acqf_discrete

from .searchspace import SearchSpace
from .utils import (
    IncompatibleSearchSpaceError,
    isabstract,
    NoMCAcquisitionFunctionError,
    to_tensor,
)


class Recommender(ABC):
    """
    Abstract base class for all recommenders.

    The job of a recommender is to select (i.e. "recommend") a subset of candidate
    experiments based on an underlying (batch) acquisition criterion.
    """

    type: str
    SUBCLASSES: Dict[str, Type[Recommender]] = {}

    compatible_discrete: ClassVar[bool]
    compatible_continuous: ClassVar[bool]

    def __init__(self, acquisition_function: Optional[AcquisitionFunction]):
        self.acquisition_function = acquisition_function

    @classmethod
    def __init_subclass__(cls, **kwargs):
        """Registers new subclasses dynamically."""
        super().__init_subclass__(**kwargs)
        if not isabstract(cls):
            cls.SUBCLASSES[cls.type] = cls

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
        # TODO[11179]: Potentially move call to get_candidates from _recommend to here
        # TODO[11179]: Potentially move metadata update from _recommend to here

        # Validate the search space
        self.check_searchspace_compatibility(searchspace)

        return self._recommend(
            searchspace,
            batch_quantity,
            allow_repeated_recommendations,
            allow_recommending_already_measured,
        )

    @abstractmethod
    def _recommend(
        self,
        searchspace: SearchSpace,
        batch_quantity: int = 1,
        allow_repeated_recommendations: bool = False,
        allow_recommending_already_measured: bool = True,
    ) -> pd.DataFrame:
        """
        Implements the actual recommendation logic. In contrast to its public
        counterpart, no validation or post-processing is carried out but only the raw
        recommendation computation is conducted.
        """

    def check_searchspace_compatibility(self, searchspace: SearchSpace) -> None:
        """
        Performs a compatibility check between the recommender type and the provided
        search space.

        Parameters
        ----------
        searchspace : SearchSpace
            The search space to check for compatibility.

        Raises
        ------
        IncompatibleSearchSpaceError
            In case the recommender is not compatible with the specified search space.
        """
        if (not self.compatible_discrete) and (not searchspace.discrete.empty):
            raise IncompatibleSearchSpaceError(
                f"The search space that was passed contained a discrete part, but the "
                f"chosen recommender of type {self.type} does not support discrete "
                f"search spaces."
            )

        if (not self.compatible_continuous) and (not searchspace.continuous.empty):
            raise IncompatibleSearchSpaceError(
                f"The search space that was passed contained a continuous part, but the"
                f" chosen recommender of type {self.type} does not support continuous "
                f"search spaces."
            )


class SequentialGreedyRecommender(Recommender):
    """
    Recommends the next set of experiments by means of sequential greedy optimization,
    i.e. using a growing set of candidates points, where each new candidate is
    selected by optimizing the joint acquisition score of the current candidate set
    while having fixed all previous candidates.

    Note: This approach only works with Monte Carlo acquisition functions, i.e. of
        type `botorch.acquisition.monte_carlo.MCAcquisitionFunction`.
    """

    # TODO: generalize the approach to also support continuous spaces

    type = "SEQUENTIAL_GREEDY"
    compatible_discrete = True
    compatible_continuous = False

    def _recommend(
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
    compatible_discrete = True
    compatible_continuous = False

    def _recommend(
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
    compatible_discrete = True
    compatible_continuous = True

    def _recommend(
        self,
        searchspace: SearchSpace,
        batch_quantity: int = 1,
        allow_repeated_recommendations: bool = False,
        allow_recommending_already_measured: bool = True,
    ) -> pd.DataFrame:
        """See base class."""

        # Discrete part if applicable
        rec_disc = pd.DataFrame()
        if not searchspace.discrete.empty:
            candidates_exp, _ = searchspace.discrete.get_candidates(
                allow_repeated_recommendations,
                allow_recommending_already_measured,
            )

            # randomly select from discrete candidates
            rec_disc = candidates_exp.sample(n=batch_quantity)
            searchspace.discrete.metadata.loc[rec_disc.index, "was_recommended"] = True

        # Continuous part if applicable
        rec_conti = pd.DataFrame()
        if not searchspace.continuous.empty:
            rec_conti = searchspace.continuous.samples_random(n_points=batch_quantity)

        # Merge sub-parts and reorder columns to match original order
        rec = pd.concat([rec_disc, rec_conti], axis=1).reindex(
            columns=[p.name for p in searchspace.parameters]
        )

        return rec


# TODO finalize class and type name below
class PurelyContinuousRecommender(Recommender):
    """
    Recommends the next set of experiments by means of sequential greedy optimization
    in a purely continuous space.
    """

    type = "CONTI"
    compatible_discrete = False
    compatible_continuous = True

    def _recommend(
        self,
        searchspace: SearchSpace,
        batch_quantity: int = 1,
        allow_repeated_recommendations: bool = False,
        allow_recommending_already_measured: bool = True,
    ) -> pd.DataFrame:
        """See base class."""

        try:
            points, _ = optimize_acqf(
                acq_function=self.acquisition_function,
                bounds=searchspace.tensor_bounds.T,
                q=batch_quantity,
                num_restarts=5,  # TODO make choice for num_restarts
                raw_samples=10,  # TODO make choice for raw_samples
            )
        except AttributeError as ex:
            raise NoMCAcquisitionFunctionError(
                f"The '{self.__class__.__name__}' only works with Monte Carlo "
                f"acquisition functions."
            ) from ex

        rec = pd.DataFrame(points, columns=searchspace.continuous.param_names)

        return rec
