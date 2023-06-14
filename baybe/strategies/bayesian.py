# pylint: disable=missing-class-docstring, missing-function-docstring
# TODO: add docstrings

"""Recommendation strategies based on Bayesian optimization."""

from typing import Callable, Optional

import pandas as pd
from attrs import define, Factory
from botorch.optim import optimize_acqf, optimize_acqf_discrete

from baybe.acquisition import PartialAcquisitionFunction

from baybe.searchspace import (
    SearchSpace,
    SearchSpaceType,
    SubspaceContinuous,
    SubspaceDiscrete,
)
from baybe.strategies.recommender import BayesianRecommender, Recommender
from baybe.utils import NoMCAcquisitionFunctionError, to_tensor


class SequentialGreedyRecommender(BayesianRecommender):

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


@define
class NaiveHybridBayesianRecommender(Recommender):

    compatibility = SearchSpaceType.HYBRID
    disc_recommender: BayesianRecommender = Factory(SequentialGreedyRecommender)
    cont_recommender: BayesianRecommender = Factory(SequentialGreedyRecommender)

    def recommend(
        self,
        searchspace: SearchSpace,
        batch_quantity: int = 1,
        train_x: Optional[pd.DataFrame] = None,
        train_y: Optional[pd.DataFrame] = None,
        allow_repeated_recommendations: bool = False,
        allow_recommending_already_measured: bool = True,
    ) -> pd.DataFrame:
        """See base class."""
        # NOTE Since this function creates new independent SearchSpace objects for
        # the discrete and continuous subspaces, it will probably not be possible
        # to have the results obey constraints that are defined for full hybrid spaces.

        # Check if the space is a pure continuous or discrete space first and just use
        # the corresponding recommendation function in that case
        degenerate_recommender = None
        if searchspace.type == SearchSpaceType.DISCRETE:
            degenerate_recommender = self.disc_recommender
        elif searchspace.type == SearchSpaceType.CONTINUOUS:
            degenerate_recommender = self.cont_recommender
        if degenerate_recommender is not None:
            return degenerate_recommender.recommend(
                searchspace=searchspace,
                batch_quantity=batch_quantity,
                train_x=train_x,
                train_y=train_y,
                allow_repeated_recommendations=allow_repeated_recommendations,
                allow_recommending_already_measured=allow_recommending_already_measured,
            )

        # We are in a hybrid setting now

        # We will attach continuous parts to discrete parts and the other way round.
        # To make things simple, we sample a single point in the continuous space which
        # will then be attached to every discrete point when the acquisition function
        # is evaluated.
        cont_part = searchspace.continuous.samples_random(1)
        cont_part = to_tensor(cont_part).unsqueeze(-2)

        # Get discrete candidates. The metadata flags are ignored since the searchspace
        # is hybrid
        # TODO Slight BOILERPLATE CODE, see recommender.py, ll. 47+
        _, candidates_comp = searchspace.discrete.get_candidates(
            allow_repeated_recommendations=True,
            allow_recommending_already_measured=True,
        )

        # Get the original acquisition function of the discrete recommender
        acqf = self.disc_recommender.setup_acquisition_function(
            searchspace, train_x, train_y
        )

        # Construct the discrete space as a standalone searchspace
        discrete_subspace = SearchSpace(searchspace.discrete, SubspaceContinuous([]))
        # Construct the partial acquisition function that attaches cont_part
        # whenever evaluating the acquisition function
        disc_acqf_part = PartialAcquisitionFunction(
            acqf=acqf, pinned_part=cont_part, pin_discrete=False
        )

        # Call the private function of the discrete recommender and get the indices
        disc_rec_idx = (
            self.disc_recommender._recommend_discrete(  # pylint: disable=W0212
                disc_acqf_part, discrete_subspace, candidates_comp, batch_quantity
            )
        )

        # Get one random discrete point that will be attached when evaluating the
        # acquisition function in the discrete space.
        disc_part = searchspace.discrete.comp_rep.loc[disc_rec_idx].sample(1)
        disc_part = to_tensor(disc_part).unsqueeze(-2)

        # Setup a fresh acquisition function for the continuous recommender
        acqf = self.cont_recommender.setup_acquisition_function(
            searchspace, train_x, train_y
        )

        # Construct the continuous space as a standalone space
        continuous_subspace = SearchSpace(
            SubspaceDiscrete.create([]), searchspace.continuous
        )
        cont_acqf_part = PartialAcquisitionFunction(
            acqf=acqf, pinned_part=disc_part, pin_discrete=True
        )
        # Call the private function of the continuous recommender
        rec_cont = self.cont_recommender._recommend_continuous(  # pylint: disable=W0212
            cont_acqf_part, continuous_subspace, batch_quantity
        )

        # Glue the solutions together and return them
        rec_disc_exp = searchspace.discrete.exp_rep.loc[disc_rec_idx]
        rec_cont.index = rec_disc_exp.index
        rec_exp = pd.concat([rec_disc_exp, rec_cont], axis=1)
        return rec_exp
