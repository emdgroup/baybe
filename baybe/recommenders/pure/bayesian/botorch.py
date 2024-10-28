"""Botorch recommender."""

from __future__ import annotations

import gc
import math
from collections.abc import Collection
from typing import TYPE_CHECKING, Any, ClassVar

import pandas as pd
from attrs import define, field
from attrs.converters import optional as optional_c
from attrs.validators import ge, gt, instance_of
from typing_extensions import override

from baybe.acquisition.acqfs import qThompsonSampling
from baybe.constraints import ContinuousCardinalityConstraint
from baybe.exceptions import (
    IncompatibilityError,
    IncompatibleAcquisitionFunctionError,
)
from baybe.parameters.numerical import _FixedNumericalContinuousParameter
from baybe.recommenders.pure.bayesian.base import BayesianRecommender
from baybe.searchspace import (
    SearchSpace,
    SearchSpaceType,
    SubspaceContinuous,
    SubspaceDiscrete,
)
from baybe.utils.dataframe import to_tensor
from baybe.utils.plotting import to_string
from baybe.utils.sampling_algorithms import (
    DiscreteSamplingMethod,
    sample_numerical_df,
)

if TYPE_CHECKING:
    from torch import Tensor


@define(kw_only=True)
class BotorchRecommender(BayesianRecommender):
    """A pure recommender utilizing Botorch's optimization machinery.

    This recommender makes use of Botorch's ``optimize_acqf_discrete``,
    ``optimize_acqf`` and ``optimize_acqf_mixed`` functions to optimize discrete,
    continuous and hybrid search spaces, respectively. Accordingly, it can be applied to
    all kinds of search spaces.

    Note:
        In hybrid search spaces, the used algorithm performs a brute-force optimization
        that can be computationally expensive. Thus, the behavior of the algorithm in
        hybrid search spaces can be controlled via two additional parameters.
    """

    # Class variables
    compatibility: ClassVar[SearchSpaceType] = SearchSpaceType.HYBRID
    # See base class.

    # Object variables
    sequential_continuous: bool = field(default=False)
    """Flag defining whether to apply sequential greedy or batch optimization in
    **continuous** search spaces. In discrete/hybrid spaces, sequential greedy
    optimization is applied automatically.
    """

    hybrid_sampler: DiscreteSamplingMethod | None = field(
        converter=optional_c(DiscreteSamplingMethod), default=None
    )
    """Strategy used for sampling the discrete subspace when performing hybrid search
    space optimization."""

    sampling_percentage: float = field(default=1.0)
    """Percentage of discrete search space that is sampled when performing hybrid search
    space optimization. Ignored when ``hybrid_sampler="None"``."""

    n_restarts: int = field(validator=[instance_of(int), gt(0)], default=10)
    """Number of times gradient-based optimization is restarted from different initial
    points. **Does not affect purely discrete optimization**.
    """

    n_raw_samples: int = field(validator=[instance_of(int), gt(0)], default=64)
    """Number of raw samples drawn for the initialization heuristic in gradient-based
    optimization. **Does not affect purely discrete optimization**.
    """

    n_threshold_inactive_parameters_generator: int = field(
        default=10, validator=[instance_of(int), ge(1)]
    )
    """Threshold used for checking which inactive parameters generator is used when
    cardinality constraints are present. When the size of the combinatorial list of
    all possible inactive parameters is larger than the threshold, a fixed number of
    randomly generated inactive parameter configurations are used and the best
    optimum among them is recommended; Otherwise, we find the best one by iterating the
    combinatorial list of all possible inactive parameters """

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

    @override
    def _recommend_discrete(
        self,
        subspace_discrete: SubspaceDiscrete,
        candidates_exp: pd.DataFrame,
        batch_size: int,
    ) -> pd.Index:
        """Generate recommendations from a discrete search space.

        Args:
            subspace_discrete: The discrete subspace from which to generate
                recommendations.
            candidates_exp: The experimental representation of all discrete candidate
                points to be considered.
            batch_size: The size of the recommendation batch.

        Raises:
            IncompatibleAcquisitionFunctionError: If a non-Monte Carlo acquisition
                function is used with a batch size > 1.

        Returns:
            The dataframe indices of the recommended points in the provided
            experimental representation.
        """
        if batch_size > 1 and not self.acquisition_function.is_mc:
            raise IncompatibleAcquisitionFunctionError(
                f"The '{self.__class__.__name__}' only works with Monte Carlo "
                f"acquisition functions for batch sizes > 1."
            )
        if batch_size > 1 and isinstance(self.acquisition_function, qThompsonSampling):
            raise IncompatibilityError(
                "Thompson sampling currently only supports a batch size of 1."
            )

        from botorch.optim import optimize_acqf_discrete

        # determine the next set of points to be tested
        candidates_comp = subspace_discrete.transform(candidates_exp)
        points, _ = optimize_acqf_discrete(
            self._botorch_acqf, batch_size, to_tensor(candidates_comp)
        )

        # retrieve the index of the points from the input dataframe
        # IMPROVE: The merging procedure is conceptually similar to what
        #   `SearchSpace._match_measurement_with_searchspace_indices` does, though using
        #   a simpler matching logic. When refactoring the SearchSpace class to
        #   handle continuous parameters, a corresponding utility could be extracted.
        idxs = pd.Index(
            pd.merge(
                pd.DataFrame(points, columns=candidates_comp.columns),
                candidates_comp.reset_index(),
                on=list(candidates_comp),
                how="left",
            )["index"]
        )

        return idxs

    @override
    def _recommend_continuous(
        self,
        subspace_continuous: SubspaceContinuous,
        batch_size: int,
    ) -> pd.DataFrame:
        """Generate recommendations from a continuous search space.

        Args:
            subspace_continuous: The continuous subspace from which to generate
                recommendations.
            batch_size: The size of the recommendation batch.

        Raises:
            IncompatibleAcquisitionFunctionError: If a non-Monte Carlo acquisition
                function is used with a batch size > 1.

        Returns:
            A dataframe containing the recommendations as individual rows.
        """
        if batch_size > 1 and not self.acquisition_function.is_mc:
            raise IncompatibleAcquisitionFunctionError(
                f"The '{self.__class__.__name__}' only works with Monte Carlo "
                f"acquisition functions for batch sizes > 1."
            )

        if len(subspace_continuous.constraints_cardinality):
            points, _ = self._recommend_continuous_with_cardinality_constraints(
                subspace_continuous, batch_size
            )
        else:
            points, _ = self._recommend_continuous_without_cardinality_constraints(
                subspace_continuous, batch_size
            )

        return pd.DataFrame(points, columns=subspace_continuous.parameter_names)

    def _recommend_continuous_with_cardinality_constraints(
        self,
        subspace_continuous: SubspaceContinuous,
        batch_size: int,
    ) -> tuple[Tensor, Tensor]:
        """Recommend from a continuous search space with cardinality constraints.

        Args:
            subspace_continuous: The continuous subspace from which to generate
                recommendations.
            batch_size: The size of the recommendation batch.

        Returns:
            The recommendations and corresponding acquisition values.

        Raises:
            ValueError: If the continuous search space has no cardinality constraints.
        """
        import torch

        if not subspace_continuous.constraints_cardinality:
            raise ValueError(
                f"'{self._recommend_continuous_with_cardinality_constraints.__name__}' "
                f"expects a subspace with constraints of type "
                f"'{ContinuousCardinalityConstraint.__name__}'. "
            )

        acqf_values_all: list[Tensor] = []
        points_all: list[Tensor] = []

        def append_recommendation_for_inactive_parameters_setting(
            inactive_parameters: Collection[str],
        ):
            """Append the recommendation for each inactive parameter configuration.

            Args:
                inactive_parameters: A list of inactive parameters.
            """
            # Create a new subspace by ensuring all active parameters being
            # non-zeros.
            subspace_continuous_without_cardinality_constraints = (
                subspace_continuous._enforce_cardinality_constraints_via_assignment(
                    inactive_parameters
                )
            )
            try:
                # Optimize the acquisition function
                (
                    points_i,
                    acqf_values_i,
                ) = self._recommend_continuous_without_cardinality_constraints(
                    subspace_continuous_without_cardinality_constraints,
                    batch_size,
                )
                # Append recommendation list and acquisition function values
                points_all.append(points_i.unsqueeze(0))
                acqf_values_all.append(acqf_values_i.unsqueeze(0))

            # The optimization problem may be infeasible for certain inactive
            # parameters. The optimize_acqf raises a ValueError when the optimization
            # problem is infeasible.
            except ValueError:
                pass

        # Below we start recommendation
        if (
            subspace_continuous.n_inactive_parameter_combinations
            > self.n_threshold_inactive_parameters_generator
        ):
            # When the combinatorial list is too large, randomly set some parameters
            # inactive.
            for _ in range(self.n_threshold_inactive_parameters_generator):
                inactive_params_sample = tuple(
                    subspace_continuous._sample_inactive_parameters(1)[0]
                )
                append_recommendation_for_inactive_parameters_setting(
                    inactive_params_sample
                )
        else:
            # When the combinatorial list is not too large, iterate the combinatorial
            # list of all possible inactive parameters.
            for (
                inactive_params_generator
            ) in subspace_continuous.inactive_parameter_combinations():
                append_recommendation_for_inactive_parameters_setting(
                    inactive_params_generator
                )

        # Find the best option
        points = torch.cat(points_all)[torch.argmax(torch.cat(acqf_values_all)), :]
        acqf_values = torch.max(torch.cat(acqf_values_all))
        return points, acqf_values

    def _recommend_continuous_without_cardinality_constraints(
        self,
        subspace_continuous: SubspaceContinuous,
        batch_size: int,
    ) -> tuple[Tensor, Tensor]:
        """Recommend from a continuous search space without cardinality constraints.

        Args:
            subspace_continuous: The continuous subspace from which to generate
                recommendations.
            batch_size: The size of the recommendation batch.

        Returns:
            The recommendations and corresponding acquisition values.

        Raises:
            ValueError: If the continuous search space has cardinality constraints.
        """
        import torch
        from botorch.optim import optimize_acqf

        if subspace_continuous.constraints_cardinality:
            raise ValueError(
                f"'{self._recommend_continuous_without_cardinality_constraints.__name__}' "  # noqa: E501
                f"expects a subspace without constraints of type "
                f"'{ContinuousCardinalityConstraint.__name__}'. "
            )

        fixed_parameters = {
            idx: p.value
            for (idx, p) in enumerate(subspace_continuous.parameters)
            if isinstance(p, _FixedNumericalContinuousParameter)
        }

        points, acqf_values = optimize_acqf(
            acq_function=self._botorch_acqf,
            bounds=torch.from_numpy(subspace_continuous.comp_rep_bounds.values),
            q=batch_size,
            num_restarts=self.n_restarts,
            raw_samples=self.n_raw_samples,
            # TODO: https://github.com/pytorch/botorch/issues/2042
            fixed_features=fixed_parameters or None,
            # TODO: https://github.com/pytorch/botorch/issues/2042
            equality_constraints=[
                c.to_botorch(subspace_continuous.parameters)
                for c in subspace_continuous.constraints_lin_eq
            ]
            or None,
            # TODO: https://github.com/pytorch/botorch/issues/2042
            inequality_constraints=[
                c.to_botorch(subspace_continuous.parameters)
                for c in subspace_continuous.constraints_lin_ineq
            ]
            or None,
            sequential=self.sequential_continuous,
        )
        return points, acqf_values

    @override
    def _recommend_hybrid(
        self,
        searchspace: SearchSpace,
        candidates_exp: pd.DataFrame,
        batch_size: int,
    ) -> pd.DataFrame:
        """Recommend points using the ``optimize_acqf_mixed`` function of BoTorch.

        This functions samples points from the discrete subspace, performs optimization
        in the continuous subspace with these points being fixed and returns the best
        found solution.

        **Important**: This performs a brute-force calculation by fixing every possible
        assignment of discrete variables and optimizing the continuous subspace for
        each of them. It is thus computationally expensive.

        **Note**: This function implicitly assumes that discrete search space parts in
        the respective data frame come first and continuous parts come second.

        Args:
            searchspace: The search space in which the recommendations should be made.
            candidates_exp: The experimental representation of the candidates
                of the discrete subspace.
            batch_size: The size of the calculated batch.

        Raises:
            IncompatibleAcquisitionFunctionError: If a non-Monte Carlo acquisition
                function is used with a batch size > 1.

        Returns:
            The recommended points.
        """
        # For batch size > 1, this optimizer needs a MC acquisition function
        if batch_size > 1 and not self.acquisition_function.is_mc:
            raise IncompatibleAcquisitionFunctionError(
                f"The '{self.__class__.__name__}' only works with Monte Carlo "
                f"acquisition functions for batch sizes > 1."
            )

        import torch
        from botorch.optim import optimize_acqf_mixed

        # Transform discrete candidates
        candidates_comp = searchspace.discrete.transform(candidates_exp)

        # Calculate the number of samples from the given percentage
        n_candidates = math.ceil(self.sampling_percentage * len(candidates_comp.index))

        # Potential sampling of discrete candidates
        if self.hybrid_sampler is not None:
            candidates_comp = sample_numerical_df(
                candidates_comp, n_candidates, method=self.hybrid_sampler
            )

        # Prepare all considered discrete configurations in the
        # List[Dict[int, float]] format expected by BoTorch.
        num_comp_columns = len(candidates_comp.columns)
        candidates_comp.columns = list(range(num_comp_columns))  # type: ignore
        fixed_features_list = candidates_comp.to_dict("records")

        # Actual call of the BoTorch optimization routine
        points, _ = optimize_acqf_mixed(
            acq_function=self._botorch_acqf,
            bounds=torch.from_numpy(searchspace.comp_rep_bounds.values),
            q=batch_size,
            num_restarts=self.n_restarts,
            raw_samples=self.n_raw_samples,
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

        # Align candidates with search space index. Done via including the search space
        # index during the merge, which is used later for back-translation into the
        # experimental representation
        merged = pd.merge(
            pd.DataFrame(points),
            candidates_comp.reset_index(),
            on=list(candidates_comp.columns),
            how="left",
        ).set_index("index")

        # Get experimental representation of discrete part
        rec_disc_exp = searchspace.discrete.exp_rep.loc[merged.index]

        # Combine discrete and continuous parts
        rec_exp = pd.concat(
            [
                rec_disc_exp,
                merged.iloc[:, num_comp_columns:].set_axis(
                    searchspace.continuous.parameter_names, axis=1
                ),
            ],
            axis=1,
        )

        return rec_exp

    @override
    def __str__(self) -> str:
        fields = [
            to_string("Surrogate", self._surrogate_model),
            to_string(
                "Acquisition function", self.acquisition_function, single_line=True
            ),
            to_string("Compatibility", self.compatibility, single_line=True),
            to_string(
                "Sequential continuous", self.sequential_continuous, single_line=True
            ),
            to_string("Hybrid sampler", self.hybrid_sampler, single_line=True),
            to_string(
                "Sampling percentage", self.sampling_percentage, single_line=True
            ),
        ]
        return to_string(self.__class__.__name__, *fields)


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
