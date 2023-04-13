# pylint: disable=missing-class-docstring, missing-function-docstring
# TODO: add docstrings

"""Recommendation strategies based on Bayesian optimization."""

from typing import Callable

import pandas as pd
from botorch.optim import optimize_acqf, optimize_acqf_discrete

from baybe.searchspace import SearchSpace, SearchSpaceType
from baybe.strategies.recommender import BayesianRecommender
from baybe.utils import NoMCAcquisitionFunctionError, to_tensor


class GreedyRecommender(BayesianRecommender):

    type = "GREEDY"
    compatibility = SearchSpaceType.EITHER

    def _recommend_discrete(
        self,
        acquisition_function: Callable,
        searchspace: SearchSpace,
        candidates_comp: pd.DataFrame,
        batch_quantity: int,
    ) -> pd.Index:
        """See base class."""
        # determine the next set of points to be tested
        candidates_tensor = to_tensor(candidates_comp)
        try:
            points, _ = optimize_acqf_discrete(
                acquisition_function, batch_quantity, candidates_tensor
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

        return idxs

    def _recommend_continuous(
        self,
        acquisition_function: Callable,
        searchspace: SearchSpace,
        batch_quantity: int,
    ) -> pd.DataFrame:
        """See base class."""

        try:
            points, _ = optimize_acqf(
                acq_function=acquisition_function,
                bounds=searchspace.param_bounds_comp,
                q=batch_quantity,
                num_restarts=5,  # TODO make choice for num_restarts
                raw_samples=10,  # TODO make choice for raw_samples
            )
        except AttributeError as ex:
            raise NoMCAcquisitionFunctionError(
                f"The '{self.__class__.__name__}' only works with Monte Carlo "
                f"acquisition functions."
            ) from ex

        # Return optimized points as dataframe
        rec = pd.DataFrame(points, columns=searchspace.continuous.param_names)
        return rec
