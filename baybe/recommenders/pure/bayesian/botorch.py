"""Botorch recommender."""

from __future__ import annotations

import math
import warnings
from collections.abc import Collection, Iterable
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd
import torch
from attrs import define, field
from attrs.converters import optional as optional_c
from attrs.validators import ge, gt, instance_of
from typing_extensions import override

from baybe.acquisition.acqfs import qThompsonSampling
from baybe.constraints import ContinuousCardinalityConstraint
from baybe.constraints.utils import is_cardinality_fulfilled
from baybe.exceptions import (
    IncompatibilityError,
    IncompatibleAcquisitionFunctionError,
    InfeasibilityError,
    MinimumCardinalityViolatedWarning,
)
from baybe.objectives.base import Objective
from baybe.parameters.numerical import _FixedNumericalContinuousParameter
from baybe.recommenders.pure.bayesian.base import BayesianRecommender
from baybe.searchspace import (
    SearchSpace,
    SearchSpaceType,
    SubspaceContinuous,
    SubspaceDiscrete,
)
from baybe.utils.conversion import to_string
from baybe.utils.dataframe import to_tensor
from baybe.utils.device_utils import (
    device_context,
    get_default_device,
    to_device,
)
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
    sequential_continuous: bool = field(default=True)
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

    device: torch.device | None = field(
        default=None,
        converter=lambda x: get_default_device() if x is None else x,
    )
    """The device to use for computations. If None, uses CUDA if available, else CPU."""

    max_n_subspaces: int = field(default=10, validator=[instance_of(int), ge(1)])
    """Threshold defining the maximum number of subspaces to consider for exhaustive
    search in the presence of cardinality constraints. If the combinatorial number of
    groupings into active and inactive parameters dictated by the constraints is greater
    than this number, that many randomly selected combinations are selected for
    optimization."""

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
        assert self._objective is not None
        acqf = self._get_acquisition_function(self._objective)
        if batch_size > 1 and not acqf.supports_batching:
            raise IncompatibleAcquisitionFunctionError(
                f"The '{self.__class__.__name__}' only works with Monte Carlo "
                f"acquisition functions for batch sizes > 1."
            )
        if batch_size > 1 and isinstance(acqf, qThompsonSampling):
            raise IncompatibilityError(
                "Thompson sampling currently only supports a batch size of 1."
            )

        from botorch.optim import optimize_acqf_discrete

        # Ensure acquisition function is on the correct device if possible
        if hasattr(self._botorch_acqf, "to"):
            self._botorch_acqf = self._botorch_acqf.to(self.device)

        # Clear CUDA cache before heavy computations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # determine the next set of points to be tested
        candidates_comp = subspace_discrete.transform(candidates_exp)
        candidates_tensor = to_device(to_tensor(candidates_comp), self.device)

        points, _ = optimize_acqf_discrete(
            self._botorch_acqf, batch_size, candidates_tensor
        )

        # Ensure the output points are moved to CPU before converting to NumPy
        points = points.detach().cpu()

        # retrieve the index of the points from the input dataframe
        # IMPROVE: The merging procedure is conceptually similar to what
        #   `SearchSpace._match_measurement_with_searchspace_indices` does, though using
        #   a simpler matching logic. When refactoring the SearchSpace class to
        #   handle continuous parameters, a corresponding utility could be extracted.
        idxs = pd.Index(
            pd.merge(
                pd.DataFrame(points.numpy(), columns=candidates_comp.columns),
                candidates_comp.reset_index(),
                on=list(candidates_comp.columns),
                how="left",
            )["index"]
        )

        # Convert idxs (which is a pd.Index) to actual indices in candidates_exp
        return idxs  # This returns just the indices

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
            A pandas DataFrame containing the recommendations as individual rows.
        """
        assert self._objective is not None
        if (
            batch_size > 1
            and not self._get_acquisition_function(self._objective).supports_batching
        ):
            raise IncompatibleAcquisitionFunctionError(
                f"The '{self.__class__.__name__}' only works with Monte Carlo "
                f"acquisition functions for batch sizes > 1."
            )

        points, _ = self._recommend_continuous_torch(subspace_continuous, batch_size)

        # Before returning DataFrame, ensure points are on CPU
        if torch.cuda.is_available() and hasattr(points, "is_cuda") and points.is_cuda:
            points = points.detach().cpu()

        return pd.DataFrame(points, columns=subspace_continuous.parameter_names)

    def _recommend_continuous_torch(
        self, subspace_continuous: SubspaceContinuous, batch_size: int
    ) -> tuple[Tensor, Tensor]:
        """Dispatcher selecting the continuous optimization routine."""
        if subspace_continuous.constraints_cardinality:
            return self._recommend_continuous_with_cardinality_constraints(
                subspace_continuous, batch_size
            )
        else:
            return self._recommend_continuous_without_cardinality_constraints(
                subspace_continuous, batch_size
            )

    def _recommend_continuous_with_cardinality_constraints(
        self,
        subspace_continuous: SubspaceContinuous,
        batch_size: int,
    ) -> tuple[Tensor, Tensor]:
        """Recommend from a continuous search space with cardinality constraints.

        This is achieved by considering the individual restricted subspaces that can be
        obtained by splitting the parameters into sets of active and inactive
        parameters, according to what is allowed by the cardinality constraints.

        The specific collection of subspaces considered by the recommender is obtained
        as either the full combinatorial set of possible parameter splits or a random
        selection thereof, depending on the upper bound specified by the corresponding
        recommender attribute.

        In each of these spaces, the (in)activity assignment is fixed, so that the
        cardinality constraints can be removed and a regular optimization can be
        performed. The recommendation is then constructed from the combined optimization
        results of the unconstrained spaces.

        Args:
            subspace_continuous: The continuous subspace from which to generate
                recommendations.
            batch_size: The size of the recommendation batch.

        Returns:
            The recommendations and corresponding acquisition values.

        Raises:
            ValueError: If the continuous search space has no cardinality constraints.
        """
        if not subspace_continuous.constraints_cardinality:
            raise ValueError(
                f"'{self._recommend_continuous_with_cardinality_constraints.__name__}' "
                f"expects a subspace with constraints of type "
                f"'{ContinuousCardinalityConstraint.__name__}'. "
            )

        # Determine search scope based on number of inactive parameter combinations
        exhaustive_search = (
            subspace_continuous.n_inactive_parameter_combinations
            <= self.max_n_subspaces
        )
        iterator: Iterable[Collection[str]]
        if exhaustive_search:
            # If manageable, evaluate all combinations of inactive parameters
            iterator = subspace_continuous.inactive_parameter_combinations()
        else:
            # Otherwise, draw a random subset of inactive parameter combinations
            iterator = subspace_continuous._sample_inactive_parameters(
                self.max_n_subspaces
            )

        # Create iterable of subspaces to be optimized
        subspaces = (
            (subspace_continuous._enforce_cardinality_constraints(inactive_parameters))
            for inactive_parameters in iterator
        )

        points, acqf_value = self._optimize_continuous_subspaces(subspaces, batch_size)

        # Ensure points are on CPU for numpy/pandas operations
        if torch.cuda.is_available() and points.is_cuda:
            points = points.detach().cpu()
            acqf_value = acqf_value.detach().cpu()

        # Check if any minimum cardinality constraints are violated
        if not is_cardinality_fulfilled(
            pd.DataFrame(points, columns=subspace_continuous.parameter_names),
            subspace_continuous,
            check_maximum=False,
        ):
            warnings.warn(
                "At least one minimum cardinality constraint has been violated. "
                "This may occur when parameter ranges extend beyond zero in both "
                "directions, making the feasible region non-convex. For such "
                "parameters, minimum cardinality constraints are currently not "
                "enforced due to the complexity of the resulting optimization problem.",
                MinimumCardinalityViolatedWarning,
            )

        return points, acqf_value

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

        # Move acquisition function to the device
        if hasattr(self._botorch_acqf, "to"):
            self._botorch_acqf = self._botorch_acqf.to(self.device)

        # Get bounds as a tensor and move to the recommender's device
        bounds_tensor = to_device(
            torch.from_numpy(subspace_continuous.comp_rep_bounds.values), self.device
        )

        # Find any fixed parameters
        fixed_parameters = {
            idx: p.value
            for (idx, p) in enumerate(subspace_continuous.parameters)
            if isinstance(p, _FixedNumericalContinuousParameter)
        }

        # Process equality constraints if they exist
        equality_constraints = None
        if subspace_continuous.constraints_lin_eq:
            # Create list to hold constraint tuples
            processed_constraints = []

            for c in subspace_continuous.constraints_lin_eq:
                # Get constraint in BoTorch format
                idxs, coeffs, rhs = c.to_botorch(subspace_continuous.parameters)

                # Move the coefficients tensor to the same device as bounds
                coeffs = coeffs.to(self.device)

                processed_constraints.append((idxs, coeffs, rhs))

            equality_constraints = processed_constraints or None

        # Process inequality constraints if they exist
        inequality_constraints = None
        if subspace_continuous.constraints_lin_ineq:
            # Create list to hold constraint tuples
            processed_constraints = []

            for c in subspace_continuous.constraints_lin_ineq:
                # Get constraint in BoTorch format
                idxs, coeffs, rhs = c.to_botorch(subspace_continuous.parameters)

                # Move the coefficients tensor to the same device as bounds
                coeffs = coeffs.to(self.device)

                processed_constraints.append((idxs, coeffs, rhs))

            inequality_constraints = processed_constraints or None

        # NOTE: The explicit `or None` conversion is added as an additional safety net
        #   because it is unclear if the corresponding presence checks for these
        #   arguments is correctly implemented in all invoked BoTorch subroutines.
        #   For details: https://github.com/pytorch/botorch/issues/2042
        points, acqf_values = optimize_acqf(
            acq_function=self._botorch_acqf,
            bounds=bounds_tensor,
            q=batch_size,
            num_restarts=self.n_restarts,
            raw_samples=self.n_raw_samples,
            fixed_features=fixed_parameters or None,
            equality_constraints=equality_constraints,
            inequality_constraints=inequality_constraints,
            sequential=self.sequential_continuous,
        )

        # Move points to CPU if they're on CUDA and we need CPU tensors for testing
        if torch.cuda.is_available() and points.is_cuda:
            # Make a copy to avoid modifying the original
            points_cpu = points.detach().cpu()
            acqf_values_cpu = acqf_values.detach().cpu()
            # For testing, return CPU tensors
            if self.device == torch.device("cpu"):
                return points_cpu, acqf_values_cpu

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
        assert self._objective is not None
        if (
            batch_size > 1
            and not self._get_acquisition_function(self._objective).supports_batching
        ):
            raise IncompatibleAcquisitionFunctionError(
                f"The '{self.__class__.__name__}' only works with Monte Carlo "
                f"acquisition functions for batch sizes > 1."
            )

        import torch
        from botorch.optim import optimize_acqf_mixed

        # Before optimizing, explicitly force everything to the correct device
        if hasattr(self, "device") and self.device is not None:
            torch.cuda.empty_cache()  # Clear CUDA cache

            # Move model and acquisition function to CPU first if using CPU
            # This helps avoid mixed device tensors
            device_str = str(self.device)
            if "cpu" in device_str:
                # Force a complete reset of all CUDA tensors
                torch.cuda.empty_cache()

                # If using CPU, first move everything to CPU
                if (
                    hasattr(self._botorch_acqf, "model")
                    and self._botorch_acqf.model is not None
                ):
                    # Move model to CPU first
                    self._botorch_acqf.model = self._botorch_acqf.model.to("cpu")

                    # Reset all caches
                    if hasattr(self._botorch_acqf.model, "prediction_strategy"):
                        strat = self._botorch_acqf.model.prediction_strategy
                        if strat is not None:
                            # Reset all caches
                            for attr in dir(strat):
                                if attr.endswith("_cache") and hasattr(strat, attr):
                                    if isinstance(getattr(strat, attr), dict):
                                        setattr(strat, attr, {})
                                    else:
                                        setattr(strat, attr, None)

                    # Move all training inputs to CPU
                    if hasattr(self._botorch_acqf.model, "train_inputs"):
                        self._botorch_acqf.model.train_inputs = tuple(
                            x.to("cpu") for x in self._botorch_acqf.model.train_inputs
                        )

                    # Move targets to CPU
                    if hasattr(self._botorch_acqf.model, "train_targets"):
                        self._botorch_acqf.model.train_targets = (
                            self._botorch_acqf.model.train_targets.to("cpu")
                        )

                    # Move likelihood to CPU
                    if hasattr(self._botorch_acqf.model, "likelihood"):
                        self._botorch_acqf.model.likelihood = (
                            self._botorch_acqf.model.likelihood.to("cpu")
                        )

                # Also move the acquisition function
                self._botorch_acqf = self._botorch_acqf.to("cpu")

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

        # Convert bounds to a tensor and move to selected device
        bounds_tensor = to_device(
            torch.from_numpy(searchspace.comp_rep_bounds.values), self.device
        )

        # Process equality constraints if they exist
        equality_constraints = None
        if searchspace.continuous.constraints_lin_eq:
            # Create list to hold constraint tuples
            processed_constraints = []

            for c in searchspace.continuous.constraints_lin_eq:
                # Get constraint in BoTorch format with index offset
                idxs, coeffs, rhs = c.to_botorch(
                    searchspace.continuous.parameters,
                    idx_offset=len(candidates_comp.columns),
                )

                # Move the coefficients tensor to the same device as bounds
                coeffs = coeffs.to(self.device)

                processed_constraints.append((idxs, coeffs, rhs))

            equality_constraints = processed_constraints or None

        # Process inequality constraints if they exist
        inequality_constraints = None
        if searchspace.continuous.constraints_lin_ineq:
            # Create list to hold constraint tuples
            processed_constraints = []

            for c in searchspace.continuous.constraints_lin_ineq:
                # Get constraint in BoTorch format with index offset
                idxs, coeffs, rhs = c.to_botorch(
                    searchspace.continuous.parameters,
                    idx_offset=num_comp_columns,
                )

                # Move the coefficients tensor to the same device as bounds
                coeffs = coeffs.to(self.device)

                processed_constraints.append((idxs, coeffs, rhs))

            inequality_constraints = processed_constraints or None

        # Move acquisition function to the device
        if hasattr(self._botorch_acqf, "to"):
            self._botorch_acqf = self._botorch_acqf.to(self.device)

        # NOTE: The explicit `or None` conversion is added as an additional safety net
        #   because it is unclear if the corresponding presence checks for these
        #   arguments is correctly implemented in all invoked BoTorch subroutines.
        #   For details: https://github.com/pytorch/botorch/issues/2042
        points, _ = optimize_acqf_mixed(
            acq_function=self._botorch_acqf,
            bounds=bounds_tensor,
            q=batch_size,
            num_restarts=self.n_restarts,
            raw_samples=self.n_raw_samples,
            fixed_features_list=fixed_features_list,
            equality_constraints=equality_constraints,
            inequality_constraints=inequality_constraints,
        )

        # Ensure points are on CPU before any numpy operations or DataFrame creation
        if torch.cuda.is_available() and hasattr(points, "is_cuda") and points.is_cuda:
            points = points.detach().cpu()

        # Align candidates with search space index. Done via including the search space
        # index during the merge, which is used later for back-translation into the
        # experimental representation
        merged = pd.merge(
            pd.DataFrame(points.numpy()),  # Now safe to convert to numpy
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

    def _optimize_continuous_subspaces(
        self, subspaces: Iterable[SubspaceContinuous], batch_size: int
    ) -> tuple[Tensor, Tensor]:
        """Find the optimum candidates from multiple continuous subspaces.

        Important:
            Subspaces without feasible solutions will be silently ignored. If none of
            the subspaces has a feasible solution, an exception will be raised.

        Args:
            subspaces: The subspaces to consider for the optimization.
            batch_size: The number of points to be recommended.

        Raises:
            InfeasibilityError: If none of the subspaces has a feasible solution.

        Returns:
            The batch of candidates and the corresponding acquisition value.
        """
        import torch
        from botorch.exceptions.errors import InfeasibilityError as BoInfeasibilityError

        acqf_values_all: list[Tensor] = []
        points_all: list[Tensor] = []

        for subspace in subspaces:
            try:
                # Optimize the acquisition function
                # Note: We explicitly evaluate the acqf function for the batch because
                #   the object returned by the optimization routine may contain joint or
                #   individual acquisition values, depending on the whether sequential
                #   or joint optimization is applied
                p, _ = self._recommend_continuous_torch(subspace, batch_size)
                with torch.no_grad():
                    acqf = self._botorch_acqf(p)

                # Append optimization results
                points_all.append(p)
                acqf_values_all.append(acqf)

            # The optimization problem may be infeasible in certain subspaces
            except BoInfeasibilityError:
                pass

        if not points_all:
            raise InfeasibilityError(
                "No feasible solution could be found. Potentially the specified "
                "constraints are too restrictive, i.e. there may be too many "
                "constraints or thresholds may have been set too tightly. "
                "Considered relaxing the constraints to improve the chances "
                "of finding a feasible solution."
            )

        # Move all tensors to CPU before using numpy functions
        acqf_values_cpu = [acqf.detach().cpu() for acqf in acqf_values_all]

        # Find the best option using numpy - now with CPU tensors
        acqf_values_np = [v.numpy() for v in acqf_values_cpu]
        best_idx = np.argmax(acqf_values_np)

        points = points_all[best_idx]
        acqf_value = acqf_values_all[best_idx]

        return points, acqf_value

    @override
    def _setup_botorch_acqf(
        self,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame | None,
        pending_experiments: pd.DataFrame | None = None,
    ) -> None:
        """Set up the BoTorch acquisition function."""
        with device_context(self.device):
            super()._setup_botorch_acqf(
                searchspace, objective, measurements, pending_experiments
            )

            # After setup is complete, if we have an acquisition function,
            # recreate the prediction strategy with a forced device
            if (
                hasattr(self, "_botorch_acqf")
                and self._botorch_acqf is not None
                and hasattr(self._botorch_acqf, "model")
                and self._botorch_acqf.model is not None
            ):
                model = self._botorch_acqf.model

                # Move model to the intended device
                model = to_device(model, self.device)

                # Detach the prediction strategy so we can rebuild it
                if hasattr(model, "prediction_strategy"):
                    model.prediction_strategy = None

                # Force rebuild of prediction strategy on the correct device
                if hasattr(model, "train_inputs") and model.train_inputs:
                    try:
                        # Move all training data to the device
                        train_x = model.train_inputs[0].to(self.device)
                        train_y = model.train_targets.to(self.device)

                        # Put model in eval mode
                        model.eval()

                        # Force a forward pass to rebuild prediction strategy
                        with torch.no_grad():
                            output = model(train_x)
                            _ = model.likelihood(output)

                            # For ExactGP models, explicitly rebuild prediction strategy
                            if hasattr(model, "exact_prediction_strategy"):
                                model.prediction_strategy = (
                                    model.exact_prediction_strategy(
                                        train_x, train_y, model.likelihood
                                    )
                                )
                    except Exception as e:
                        # If there's an error rebuilding, just log it and continue
                        print(f"Error rebuilding prediction strategy: {e}")

                # Update the model in the acquisition function
                self._botorch_acqf.model = model

                # Move the entire acquisition function to the device
                self._botorch_acqf = to_device(self._botorch_acqf, self.device)
