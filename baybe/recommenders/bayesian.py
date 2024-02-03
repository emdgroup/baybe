"""Different recommendation strategies that are based on Bayesian optimization."""

import warnings
from abc import ABC
from functools import partial
from typing import Any, Callable, ClassVar, Literal, Optional

import numpy as np
import pandas as pd
from attrs import define, evolve, field, fields, validators
from botorch.acquisition import (
    AcquisitionFunction,
    ExpectedImprovement,
    PosteriorMean,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
    qExpectedImprovement,
    qProbabilityOfImprovement,
    qUpperConfidenceBound,
)
from botorch.optim import optimize_acqf, optimize_acqf_discrete, optimize_acqf_mixed
from sklearn.metrics import pairwise_distances_argmin

from baybe.acquisition import PartialAcquisitionFunction, debotorchize
from baybe.exceptions import NoMCAcquisitionFunctionError
from baybe.recommenders.base import (
    NonPredictiveRecommender,
    Recommender,
    _select_candidates_and_recommend,
)
from baybe.searchspace import (
    SearchSpace,
    SearchSpaceType,
    SubspaceContinuous,
    SubspaceDiscrete,
)
from baybe.surrogates import _ONNX_INSTALLED, GaussianProcessSurrogate
from baybe.surrogates.base import Surrogate
from baybe.utils.dataframe import to_tensor
from baybe.utils.sampling_algorithms import farthest_point_sampling

if _ONNX_INSTALLED:
    from baybe.surrogates import CustomONNXSurrogate


@define
class BayesianRecommender(Recommender, ABC):
    """An abstract class for Bayesian Recommenders."""

    surrogate_model: Surrogate = field(factory=GaussianProcessSurrogate)
    """The used surrogate model."""

    acquisition_function_cls: Literal[
        "PM", "PI", "EI", "UCB", "qPI", "qEI", "qUCB", "VarUCB", "qVarUCB"
    ] = field(default="qEI")
    """The used acquisition function class."""

    _acquisition_function: Optional[AcquisitionFunction] = field(
        default=None, init=False
    )
    """The current acquisition function."""

    def _get_acquisition_function_cls(
        self,
    ) -> Callable:
        """Get the actual acquisition function class.

        Returns:
            The debotorchized acquisition function class.
        """
        mapping = {
            "PM": PosteriorMean,
            "PI": ProbabilityOfImprovement,
            "EI": ExpectedImprovement,
            "UCB": partial(UpperConfidenceBound, beta=1.0),
            "qEI": qExpectedImprovement,
            "qPI": qProbabilityOfImprovement,
            "qUCB": partial(qUpperConfidenceBound, beta=1.0),
            "VarUCB": partial(UpperConfidenceBound, beta=100.0),
            "qVarUCB": partial(qUpperConfidenceBound, beta=100.0),
        }
        fun = debotorchize(mapping[self.acquisition_function_cls])
        return fun

    def setup_acquisition_function(
        self, searchspace: SearchSpace, train_x: pd.DataFrame, train_y: pd.DataFrame
    ) -> None:
        """Create the current acquisition function from provided training data.

        The acquisition function is stored in the private attribute
        ``_acquisition_function``.

        Args:
            searchspace: The search space in which the experiments are to be conducted.
            train_x: The features of the conducted experiments.
            train_y: The corresponding response values.
        """
        best_f = train_y.max()
        surrogate_model = self._fit(searchspace, train_x, train_y)
        acquisition_function_cls = self._get_acquisition_function_cls()

        self._acquisition_function = acquisition_function_cls(surrogate_model, best_f)

    def _fit(
        self,
        searchspace: SearchSpace,
        train_x: pd.DataFrame,
        train_y: pd.DataFrame,
    ) -> Surrogate:
        """Train a fresh surrogate model instance for the DOE strategy.

        Args:
            searchspace: The search space.
            train_x: The features of the conducted experiments.
            train_y: The corresponding response values.

        Returns:
            A surrogate model fitted to the provided data.

        Raises:
            ValueError: If the training inputs and targets do not have the same index.
        """
        # validate input
        if not train_x.index.equals(train_y.index):
            raise ValueError("Training inputs and targets must have the same index.")

        self.surrogate_model.fit(searchspace, *to_tensor(train_x, train_y))

        return self.surrogate_model

    def recommend(  # noqa: D102
        self,
        searchspace: SearchSpace,
        batch_size: int = 1,
        train_x: Optional[pd.DataFrame] = None,
        train_y: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        # See base class.

        if _ONNX_INSTALLED and isinstance(self.surrogate_model, CustomONNXSurrogate):
            CustomONNXSurrogate.validate_compatibility(searchspace)

        self.setup_acquisition_function(searchspace, train_x, train_y)

        if searchspace.type == SearchSpaceType.DISCRETE:
            return _select_candidates_and_recommend(
                searchspace,
                self._recommend_discrete,
                batch_size,
                self.allow_repeated_recommendations,
                self.allow_recommending_already_measured,
            )
        if searchspace.type == SearchSpaceType.CONTINUOUS:
            return self._recommend_continuous(searchspace.continuous, batch_size)
        return self._recommend_hybrid(searchspace, batch_size)

    def _recommend_discrete(
        self,
        subspace_discrete: SubspaceDiscrete,
        candidates_comp: pd.DataFrame,
        batch_size: int,
    ) -> pd.Index:
        """Calculate recommendations in a discrete search space.

        Args:
            subspace_discrete: The discrete subspace in which the recommendations
                should be made.
            candidates_comp: The computational representation of all possible
                candidates.
            batch_size: The size of the calculated batch.

        Raises:
            NotImplementedError: If the function is not implemented by the child class.

        Returns:
            The indices of the recommended points with respect to the
            computational representation.
        """
        raise NotImplementedError()

    def _recommend_continuous(
        self,
        subspace_continuous: SubspaceContinuous,
        batch_size: int,
    ) -> pd.DataFrame:
        """Calculate recommendations in a continuous search space.

        Args:
            subspace_continuous: The continuous subspace in which the recommendations
                should be made.
            batch_size: The size of the calculated batch.

        Raises:
            NotImplementedError: If the function is not implemented by the child class.

        Returns:
            The recommended points.
        """
        raise NotImplementedError()

    def _recommend_hybrid(
        self,
        searchspace: SearchSpace,
        batch_size: int,
    ) -> pd.DataFrame:
        """Calculate recommendations in a hybrid search space.

        Args:
            searchspace: The hybrid search space in which the recommendations should
                be made.
            batch_size: The size of the calculated batch.

        Raises:
            NotImplementedError: If the function is not implemented by the child class.

        Returns:
            The recommended points.
        """
        raise NotImplementedError()


@define
class SequentialGreedyRecommender(BayesianRecommender):
    """Recommender using sequential Greedy optimization.

    This recommender implements the BoTorch functions ``optimize_acqf_discrete``,
    ``optimize_acqf`` and ``optimize_acqf_mixed`` for the optimization of discrete,
    continuous and hybrid search spaces. In particular, it can be applied in all
    kinds of search spaces.
    It is important to note that this algorithm performs a brute-force optimization in
    hybrid search spaces which can be computationally expensive. Thus, the behavior of
    the algorithm in hybrid search spaces can be controlled by two additional
    parameters.
    """

    # Class variables
    compatibility: ClassVar[SearchSpaceType] = SearchSpaceType.HYBRID
    # See base class.

    # Object variables
    hybrid_sampler: str = field(
        validator=validators.in_(["None", "Farthest", "Random"]), default="None"
    )
    """Strategy used for sampling the discrete subspace when performing hybrid search
    space optimization."""

    sampling_percentage: float = field(default=1.0)
    """Percentage of discrete search space that is sampled when performing hybrid search
    space optimization. Ignored when ``hybrid_sampler="None"``."""

    @sampling_percentage.validator
    def _validate_percentage(  # noqa: DOC101, DOC103
        self, _: Any, value: float
    ) -> None:
        """Validate that the given value is in fact a percentage.

        Raises:
            ValueError: If ``value`` is not between 0 and 1.
        """
        if not 0 <= value <= 1:
            raise ValueError(
                f"Hybrid sampling percentage needs to be between 0 and 1 but is {value}"
            )

    def _recommend_discrete(
        self,
        subspace_discrete: SubspaceDiscrete,
        candidates_comp: pd.DataFrame,
        batch_size: int,
    ) -> pd.Index:
        # See base class.

        # determine the next set of points to be tested
        candidates_tensor = to_tensor(candidates_comp)
        try:
            points, _ = optimize_acqf_discrete(
                self._acquisition_function, batch_size, candidates_tensor
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
        subspace_continuous: SubspaceContinuous,
        batch_size: int,
    ) -> pd.DataFrame:
        # See base class.

        try:
            points, _ = optimize_acqf(
                acq_function=self._acquisition_function,
                bounds=subspace_continuous.param_bounds_comp,
                q=batch_size,
                num_restarts=5,  # TODO make choice for num_restarts
                raw_samples=10,  # TODO make choice for raw_samples
                equality_constraints=[
                    c.to_botorch(subspace_continuous.parameters)
                    for c in subspace_continuous.constraints_lin_eq
                ]
                or None,  # TODO: https://github.com/pytorch/botorch/issues/2042
                inequality_constraints=[
                    c.to_botorch(subspace_continuous.parameters)
                    for c in subspace_continuous.constraints_lin_ineq
                ]
                or None,  # TODO: https://github.com/pytorch/botorch/issues/2042
            )
        except AttributeError as ex:
            raise NoMCAcquisitionFunctionError(
                f"The '{self.__class__.__name__}' only works with Monte Carlo "
                f"acquisition functions."
            ) from ex

        # Return optimized points as dataframe
        rec = pd.DataFrame(points, columns=subspace_continuous.param_names)
        return rec

    def _recommend_hybrid(
        self,
        searchspace: SearchSpace,
        batch_size: int,
    ) -> pd.DataFrame:
        """Recommend points using the ``optimize_acqf_mixed`` function of BoTorch.

        This functions samples points from the discrete subspace, performs optimization
        in the continuous subspace with these points being fixed and returns the best
        found solution.
        **Important**: This performs a brute-force calculation by fixing every possible
        assignment of discrete variables and optimizing the continuous subspace for
        each of them. It is thus computationally expensive.

        Args:
            searchspace: The search space in which the recommendations should be made.
            batch_size: The size of the calculated batch.

        Returns:
            The recommended points.

        Raises:
            NoMCAcquisitionFunctionError: If a non Monte Carlo acquisition function
                is chosen.
        """
        # Get discrete candidates.
        _, candidates_comp = searchspace.discrete.get_candidates(
            allow_repeated_recommendations=True,
            allow_recommending_already_measured=True,
        )

        # Calculate the number of samples from the given percentage
        n_candidates = int(self.sampling_percentage * len(candidates_comp.index))

        # Potential sampling of discrete candidates
        if self.hybrid_sampler == "Farthest":
            ilocs = farthest_point_sampling(candidates_comp.values, n_candidates)
            candidates_comp = candidates_comp.iloc[ilocs]
        elif self.hybrid_sampler == "Random":
            candidates_comp = candidates_comp.sample(n_candidates)

        # Prepare all considered discrete configurations in the List[Dict[int, float]]
        # format expected by BoTorch
        # TODO: Currently assumes that discrete parameters are first and continuous
        #   second. Once parameter redesign [11611] is completed, we might adjust this.
        candidates_comp.columns = list(range(len(candidates_comp.columns)))
        fixed_features_list = candidates_comp.to_dict("records")

        # Actual call of the BoTorch optimization routine
        try:
            points, _ = optimize_acqf_mixed(
                acq_function=self._acquisition_function,
                bounds=searchspace.param_bounds_comp,
                q=batch_size,
                num_restarts=5,  # TODO make choice for num_restarts
                raw_samples=10,  # TODO make choice for raw_samples
                fixed_features_list=fixed_features_list,
                equality_constraints=[
                    c.to_botorch(
                        searchspace.continuous.parameters,
                        idx_offset=len(candidates_comp.columns),
                    )
                    for c in searchspace.continuous.constraints_lin_eq
                ]
                or None,  # TODO: https://github.com/pytorch/botorch/issues/2042
                inequality_constraints=[
                    c.to_botorch(
                        searchspace.continuous.parameters,
                        idx_offset=len(candidates_comp.columns),
                    )
                    for c in searchspace.continuous.constraints_lin_ineq
                ]
                or None,  # TODO: https://github.com/pytorch/botorch/issues/2042
            )
        except AttributeError as ex:
            raise NoMCAcquisitionFunctionError(
                f"The '{self.__class__.__name__}' only works with Monte Carlo "
                f"acquisition functions."
            ) from ex

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # TODO [14819]: The following code is necessary due to floating point
        #   inaccuracies introduced by BoTorch (potentially due to some float32
        #   conversion?). The current workaround is the match the recommendations back
        #   to the closest candidate points.

        # Split discrete and continuous parts
        disc_points = points[:, : len(candidates_comp.columns)]
        cont_points = points[:, len(candidates_comp.columns) :]

        # Find the closest match with the discrete candidates
        candidates_comp_np = candidates_comp.to_numpy()
        disc_points_np = disc_points.numpy()
        if not disc_points_np.flags["C_CONTIGUOUS"]:
            disc_points_np = np.ascontiguousarray(disc_points_np)
        if not candidates_comp_np.flags["C_CONTIGUOUS"]:
            candidates_comp_np = np.ascontiguousarray(candidates_comp_np)
        disc_idxs_iloc = pairwise_distances_argmin(
            disc_points_np, candidates_comp_np, metric="manhattan"
        )

        # Get the actual search space dataframe indices
        disc_idxs_loc = candidates_comp.iloc[disc_idxs_iloc].index

        # Get experimental representation of discrete and continuous parts
        rec_disc_exp = searchspace.discrete.exp_rep.loc[disc_idxs_loc]
        rec_cont_exp = pd.DataFrame(
            cont_points, columns=searchspace.continuous.param_names
        )

        # Adjust the index of the continuous part and concatenate both
        rec_cont_exp.index = rec_disc_exp.index
        rec_exp = pd.concat([rec_disc_exp, rec_cont_exp], axis=1)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        return rec_exp


@define
class NaiveHybridRecommender(Recommender):
    """Recommend points by independent optimization of subspaces.

    This recommender splits the hybrid search space in the discrete and continuous
    subspace. Each of the subspaces is optimized on its own, and the recommenders for
    those subspaces can be chosen upon initialization. If this recommender is used on
    a non-hybrid space, it uses the corresponding recommender.
    """

    # TODO: This class (and potentially the recommender function signatures) need to
    #   be refactored such that there is no more coupling to BayesianRecommender and it
    #   can be moved to recommender.py

    # Class variables
    compatibility: ClassVar[SearchSpaceType] = SearchSpaceType.HYBRID
    # See base class.

    # Object variables
    # TODO This used to be a Union of BayesianRecommender and NonPredictiveRecommender.
    # Due to serialization issues, this was changed to Recommender in general.
    # As we currently do not have other subclasses of Recommender, this solution works
    # for now. Still, we manually check whether the disc_recommender belogns to one of
    # these two subclasses such that we might be able to easily spot a potential problem
    # that might come up when implementing new subclasses of Recommender
    disc_recommender: Recommender = field(factory=SequentialGreedyRecommender)
    """The recommender used for the discrete subspace. Default:
    :class:`baybe.recommenders.bayesian.SequentialGreedyRecommender`"""

    cont_recommender: BayesianRecommender = field(factory=SequentialGreedyRecommender)
    """The recommender used for the continuous subspace. Default:
    :class:`baybe.recommenders.bayesian.SequentialGreedyRecommender`"""

    def __attrs_post_init__(self):
        """Validate if flags are synchronized and overrides them otherwise."""
        if (
            flag := self.allow_recommending_already_measured
        ) != self.disc_recommender.allow_recommending_already_measured:
            warnings.warn(
                f"The value of "
                f"'{fields(self.__class__).allow_recommending_already_measured.name}' "
                f"differs from what is specified in the discrete recommender. "
                f"The value of the discrete recommender will be ignored."
            )
            self.disc_recommender = evolve(
                self.disc_recommender,
                allow_recommending_already_measured=flag,
            )

        if (
            flag := self.allow_repeated_recommendations
        ) != self.disc_recommender.allow_repeated_recommendations:
            warnings.warn(
                f"The value of "
                f"'{fields(self.__class__).allow_repeated_recommendations.name}' "
                f"differs from what is specified in the discrete recommender. "
                f"The value of the discrete recommender will be ignored."
            )
            self.disc_recommender = evolve(
                self.disc_recommender,
                allow_repeated_recommendations=flag,
            )

    def recommend(  # noqa: D102
        self,
        searchspace: SearchSpace,
        batch_size: int = 1,
        train_x: Optional[pd.DataFrame] = None,
        train_y: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        # See base class.

        # First check whether the disc_recommender is either bayesian or non-predictive
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
                batch_size=batch_size,
                train_x=train_x,
                train_y=train_y,
            )

        # We are in a hybrid setting now

        # We will attach continuous parts to discrete parts and the other way round.
        # To make things simple, we sample a single point in the continuous space which
        # will then be attached to every discrete point when the acquisition function
        # is evaluated.
        cont_part = searchspace.continuous.samples_random(1)
        cont_part = to_tensor(cont_part).unsqueeze(-2)

        # Get discrete candidates. The metadata flags are ignored since the search space
        # is hybrid
        # TODO Slight BOILERPLATE CODE, see recommender.py, ll. 47+
        _, candidates_comp = searchspace.discrete.get_candidates(
            allow_repeated_recommendations=True,
            allow_recommending_already_measured=True,
        )

        # We now check whether the discrete recommender is bayesian.
        if is_bayesian_recommender:
            # Get access to the recommenders acquisition function
            self.disc_recommender.setup_acquisition_function(
                searchspace, train_x, train_y
            )

            # Construct the partial acquisition function that attaches cont_part
            # whenever evaluating the acquisition function
            disc_acqf_part = PartialAcquisitionFunction(
                acqf=self.disc_recommender._acquisition_function,
                pinned_part=cont_part,
                pin_discrete=False,
            )

            self.disc_recommender._acquisition_function = disc_acqf_part

        # Call the private function of the discrete recommender and get the indices
        disc_rec_idx = self.disc_recommender._recommend_discrete(
            subspace_discrete=searchspace.discrete,
            candidates_comp=candidates_comp,
            batch_size=batch_size,
        )

        # Get one random discrete point that will be attached when evaluating the
        # acquisition function in the discrete space.
        disc_part = searchspace.discrete.comp_rep.loc[disc_rec_idx].sample(1)
        disc_part = to_tensor(disc_part).unsqueeze(-2)

        # Setup a fresh acquisition function for the continuous recommender
        self.cont_recommender.setup_acquisition_function(searchspace, train_x, train_y)

        # Construct the continuous space as a standalone space
        cont_acqf_part = PartialAcquisitionFunction(
            acqf=self.cont_recommender._acquisition_function,
            pinned_part=disc_part,
            pin_discrete=True,
        )
        self.cont_recommender._acquisition_function = cont_acqf_part

        # Call the private function of the continuous recommender
        rec_cont = self.cont_recommender._recommend_continuous(
            searchspace.continuous, batch_size
        )

        # Glue the solutions together and return them
        rec_disc_exp = searchspace.discrete.exp_rep.loc[disc_rec_idx]
        rec_cont.index = rec_disc_exp.index
        rec_exp = pd.concat([rec_disc_exp, rec_cont], axis=1)
        return rec_exp
