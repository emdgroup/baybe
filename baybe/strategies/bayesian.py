# pylint: disable=missing-class-docstring, missing-function-docstring
# TODO: add docstrings

"""Recommendation strategies based on Bayesian optimization."""

from typing import Callable, Optional

import numpy as np
import pandas as pd
from attrs import define, Factory, field, validators
from botorch.optim import optimize_acqf, optimize_acqf_discrete, optimize_acqf_mixed
from sklearn.metrics import pairwise_distances_argmin

from baybe.acquisition import PartialAcquisitionFunction

from baybe.searchspace import SearchSpace, SearchSpaceType
from baybe.strategies.recommender import (
    BayesianRecommender,
    NonPredictiveRecommender,
    Recommender,
)
from baybe.utils import NoMCAcquisitionFunctionError, to_tensor
from baybe.utils.sampling_algorithms import farthest_point_sampling


# Validation for the sampling percentage of the hybrid recommendation functionality.
# Modeled in the same way as validation is done in the parameters.py file
def validate_percentage(obj, _, value):
    """Validate that the given value is a proper percentage between 0 and 1."""
    if not 0 <= value <= 1:
        raise ValueError(
            f"Percentage for {obj.__class__.__name__} needs to be between 0 and 1 but "
            f"is {value}"
        )


@define
class SequentialGreedyRecommender(BayesianRecommender):

    compatibility = SearchSpaceType.HYBRID
    # Keyword for which sampling strategy should be used for hybrid recommendation
    hybrid_sampler: str = field(
        validator=validators.in_(["None", "Farthest", "Random"]), default="None"
    )
    # Percentage for how of the search space should be sampled for hybrid recommendation
    sampling_percentage: float = field(
        validator=[validators.instance_of(float), validate_percentage], default=1.0
    )

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
                bounds=searchspace.continuous.param_bounds_comp,
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

    def _recommend_hybrid(
        self,
        acquisition_function: Callable,
        searchspace: SearchSpace,
        batch_quantity: int,
    ):
        """Recommendation functionality of the SequentialGreedy Recommender.
        This implements the `optimize_acqf_mixed` function of BoTorch.

        Important: This performs a brute-force calculation by fixing evey possible
        assignment of discrete variables and optimizing the continuous subspace for
        each of them. It is thus computationally expensive."""

        # Get discrete candidates.
        _, candidates_comp = searchspace.discrete.get_candidates(
            allow_repeated_recommendations=True,
            allow_recommending_already_measured=True,
        )

        # Calculate the number of samples from the given percentage
        number_of_samples = int(self.sampling_percentage * len(candidates_comp.index))

        # Potential sampling of discrete candidates
        if self.hybrid_sampler == "Farthest":
            ilocs = farthest_point_sampling(candidates_comp.values, number_of_samples)
            candidates_comp = candidates_comp.iloc[ilocs]
        elif self.hybrid_sampler == "Random":
            candidates_comp = candidates_comp.sample(number_of_samples)

        # Since the format for the BoTorch function needs to be List[Dict[int, float]],
        # we need to get the indices of the features
        # TODO Currently assumes that discrete parameters are first and continuous
        # second. Once parameter redesign [11611] is implemented, we might adjust this.

        candidates_comp.columns = list(range(len(candidates_comp.columns)))
        fixed_features_list = candidates_comp.to_dict("records")

        # Actual call of the botorch optimization routine
        try:
            points, _ = optimize_acqf_mixed(
                acq_function=acquisition_function,
                bounds=searchspace.param_bounds_comp,
                q=batch_quantity,
                num_restarts=5,  # TODO make choice for num_restarts
                raw_samples=10,  # TODO make choice for raw_samples
                fixed_features_list=fixed_features_list,
            )
        except AttributeError as ex:
            raise NoMCAcquisitionFunctionError(
                f"The '{self.__class__.__name__}' only works with Monte Carlo "
                f"acquisition functions."
            ) from ex

        # Transform into experimental representation

        disc_points = points[:, : len(candidates_comp.columns)]
        cont_points = points[:, len(candidates_comp.columns) :]

        # Calculate the indices of the discrete part to extract the experimental
        # representation. This is done by finding closests rows in candidates_comp.
        # TODO This is currently necessary due to BoTorch changing data types internally
        # Might thus change this once [14819] is fixed.

        candidates_comp_np = candidates_comp.to_numpy()
        disc_points_np = disc_points.numpy()

        # Make everything contiguous if necessary
        if not disc_points_np.flags["C_CONTIGUOUS"]:
            disc_points_np = np.ascontiguousarray(disc_points_np)
        if not candidates_comp_np.flags["C_CONTIGUOUS"]:
            candidates_comp_np = np.ascontiguousarray(candidates_comp_np)
        disc_idxs_loc = pairwise_distances_argmin(
            disc_points_np, candidates_comp_np, metric="manhattan"
        )

        # Get actual indices, as disc_idx_loc is location based
        disc_idxs = candidates_comp.iloc[disc_idxs_loc].index

        # Get experimental representation of discrete and continuous parts
        rec_disc_exp = searchspace.discrete.exp_rep.loc[disc_idxs]
        rec_cont_exp = pd.DataFrame(
            cont_points, columns=searchspace.continuous.param_names
        )

        # Adjust index, concatenate and return
        rec_cont_exp.index = rec_disc_exp.index
        rec_exp = pd.concat([rec_disc_exp, rec_cont_exp], axis=1)

        return rec_exp


@define
class NaiveHybridRecommender(Recommender):

    compatibility = SearchSpaceType.HYBRID
    # TODO This used to be a Union of BayesianRecommender and NonPredictiveRecommender.
    # Due to serialization issues, this was changed to Recommender in general.
    # As we currently do not have other subclasses of Recommender, this solution works
    # for now. Still, we manually check whether the disc_recommender belogns to one of
    # these two subclasses such that we might be able to easily spot a potential problem
    # that might come up when implementing new subclasses of Recommender
    disc_recommender: Recommender = Factory(SequentialGreedyRecommender)
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

        # First check whether the disc_recommender is either bayesian or non predictive
        is_bayesian_recommender = isinstance(self.disc_recommender, BayesianRecommender)
        is_np_recommender = isinstance(self.disc_recommender, NonPredictiveRecommender)

        if (not is_bayesian_recommender) and (not is_np_recommender):
            raise NotImplementedError(
                """The discrete recommender should be either a Bayesian or a
                NonPredictiveRecommender."""
            )

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

        # Due to different signatures depending on whether the discrete recommender is
        # bayesian or non-predictive, we need to check what kind of recommender we have
        # This is then used to potentially fill the dictionary containing the
        # corresponding keyword and acquisition function.
        acqf_func_dict = {}
        # We now check whether the discrete recommender is bayesian.
        if is_bayesian_recommender:
            # Get access to the recommenders acquisition function
            disc_acqf = self.disc_recommender.setup_acquisition_function(
                searchspace, train_x, train_y
            )

            # Construct the partial acquisition function that attaches cont_part
            # whenever evaluating the acquisition function
            disc_acqf_part = PartialAcquisitionFunction(
                acqf=disc_acqf, pinned_part=cont_part, pin_discrete=False
            )
            acqf_func_dict = {"acquisition_function": disc_acqf_part}

        # Call the private function of the discrete recommender and get the indices
        disc_rec_idx = (
            self.disc_recommender._recommend_discrete(  # pylint: disable=W0212
                **(acqf_func_dict),
                searchspace=searchspace,
                candidates_comp=candidates_comp,
                batch_quantity=batch_quantity,
            )
        )

        # Get one random discrete point that will be attached when evaluating the
        # acquisition function in the discrete space.
        disc_part = searchspace.discrete.comp_rep.loc[disc_rec_idx].sample(1)
        disc_part = to_tensor(disc_part).unsqueeze(-2)

        # Setup a fresh acquisition function for the continuous recommender
        cont_acqf = self.cont_recommender.setup_acquisition_function(
            searchspace, train_x, train_y
        )

        # Construct the continuous space as a standalone space
        cont_acqf_part = PartialAcquisitionFunction(
            acqf=cont_acqf, pinned_part=disc_part, pin_discrete=True
        )
        # Call the private function of the continuous recommender
        rec_cont = self.cont_recommender._recommend_continuous(  # pylint: disable=W0212
            cont_acqf_part, searchspace, batch_quantity
        )

        # Glue the solutions together and return them
        rec_disc_exp = searchspace.discrete.exp_rep.loc[disc_rec_idx]
        rec_cont.index = rec_disc_exp.index
        rec_exp = pd.concat([rec_disc_exp, rec_cont], axis=1)
        return rec_exp
