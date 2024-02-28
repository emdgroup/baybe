"""Sequential greedy recommender."""

from typing import Any, ClassVar

import pandas as pd
from attrs import define, field, validators
from botorch.optim import optimize_acqf, optimize_acqf_discrete, optimize_acqf_mixed

from baybe.exceptions import NoMCAcquisitionFunctionError
from baybe.recommenders.pure.bayesian.base import BayesianRecommender
from baybe.searchspace import (
    SearchSpace,
    SearchSpaceType,
    SubspaceContinuous,
    SubspaceDiscrete,
)
from baybe.utils.dataframe import to_tensor
from baybe.utils.sampling_algorithms import farthest_point_sampling


@define
class SequentialGreedyRecommender(BayesianRecommender):
    """Pure recommender using sequential greedy optimization.

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
        candidates_comp: pd.DataFrame,
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
            candidates_comp: The computational representation of the candidates
                of the discrete subspace.
            batch_size: The size of the calculated batch.

        Returns:
            The recommended points.

        Raises:
            NoMCAcquisitionFunctionError: If a non Monte Carlo acquisition function
                is chosen.
        """
        if len(candidates_comp) > 0:
            # Calculate the number of samples from the given percentage
            n_candidates = int(self.sampling_percentage * len(candidates_comp.index))

            # Potential sampling of discrete candidates
            if self.hybrid_sampler == "Farthest":
                ilocs = farthest_point_sampling(candidates_comp.values, n_candidates)
                candidates_comp = candidates_comp.iloc[ilocs]
            elif self.hybrid_sampler == "Random":
                candidates_comp = candidates_comp.sample(n_candidates)

            # Prepare all considered discrete configurations in the
            # List[Dict[int, float]] format expected by BoTorch.
            # TODO: Currently assumes that discrete parameters are first and continuous
            #   second. Once parameter redesign [11611] is completed, we might adjust
            #   this.
            num_comp_columns = len(candidates_comp.columns)
            candidates_comp.columns = list(range(num_comp_columns))  # type: ignore
            fixed_features_list = candidates_comp.to_dict("records")
        else:
            fixed_features_list = None

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
                        idx_offset=num_comp_columns,
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

        disc_points = points[:, :num_comp_columns]
        cont_points = points[:, num_comp_columns:]

        # Get selected candidate indices
        idxs = pd.Index(
            pd.merge(
                candidates_comp.reset_index(),
                pd.DataFrame(disc_points, columns=candidates_comp.columns),
                on=list(candidates_comp),
            )["index"]
        )

        # Get experimental representation of discrete and continuous parts
        rec_disc_exp = searchspace.discrete.exp_rep.loc[idxs]
        rec_cont_exp = pd.DataFrame(
            cont_points, columns=searchspace.continuous.param_names
        )

        # Adjust the index of the continuous part and create overall recommendations
        rec_cont_exp.index = rec_disc_exp.index
        rec_exp = pd.concat([rec_disc_exp, rec_cont_exp], axis=1)

        return rec_exp
