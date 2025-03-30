"""Botorch recommender."""

import gc
import math
from typing import Any, ClassVar

import pandas as pd
import torch
from attrs import define, field
from attrs.converters import optional as optional_c
from attrs.validators import gt, instance_of
from typing_extensions import override

from baybe.acquisition.acqfs import qThompsonSampling
from baybe.exceptions import (
    IncompatibilityError,
    IncompatibleAcquisitionFunctionError,
)
from baybe.objectives.base import Objective
from baybe.recommenders.pure.bayesian.base import BayesianRecommender
from baybe.searchspace import (
    SearchSpace,
    SearchSpaceType,
    SubspaceContinuous,
    SubspaceDiscrete,
)
from baybe.utils.dataframe import to_tensor
from baybe.utils.device_mode import single_device_mode
from baybe.utils.plotting import to_string
from baybe.utils.sampling_algorithms import (
    DiscreteSamplingMethod,
    sample_numerical_df,
)


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

    device: torch.device | None = field(
        default=None,
        converter=lambda x: torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if x is None
        else x,
    )
    """The device to use for computations. If None, uses CUDA if available, else CPU."""

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

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensors to the specified device.

        Args:
            tensor: The tensor to move to the device.

        Returns:
            The tensor on the specified device.
        """
        # If we're using CPU, first clear CUDA cache to avoid memory leaks
        if self.device == torch.device("cpu") and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return tensor.to(self.device) if self.device is not None else tensor

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
        if batch_size > 1 and not self.acquisition_function.supports_batching:
            raise IncompatibleAcquisitionFunctionError(
                f"The '{self.__class__.__name__}' only works with Monte Carlo "
                f"acquisition functions for batch sizes > 1."
            )
        if batch_size > 1 and isinstance(self.acquisition_function, qThompsonSampling):
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
        candidates_tensor = self._to_device(to_tensor(candidates_comp))

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
            A dataframe containing the recommendations as individual rows.
        """
        if batch_size > 1 and not self.acquisition_function.supports_batching:
            raise IncompatibleAcquisitionFunctionError(
                f"The '{self.__class__.__name__}' only works with Monte Carlo "
                f"acquisition functions for batch sizes > 1."
            )

        import torch
        from botorch.optim import optimize_acqf

        # Move acquisition function to the device
        if hasattr(self._botorch_acqf, "to"):
            self._botorch_acqf = self._botorch_acqf.to(self.device)

        # Get bounds as a tensor and move to the recommender's device
        bounds_tensor = self._to_device(
            torch.from_numpy(subspace_continuous.comp_rep_bounds.values)
        )

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

        # Perform optimization
        points, _ = optimize_acqf(
            acq_function=self._botorch_acqf,
            bounds=bounds_tensor,
            q=batch_size,
            num_restarts=self.n_restarts,
            raw_samples=self.n_raw_samples,
            equality_constraints=equality_constraints,
            inequality_constraints=inequality_constraints,
            sequential=self.sequential_continuous,
        )

        # Move output points to CPU before returning as DataFrame
        rec = pd.DataFrame(
            points.cpu().numpy(), columns=subspace_continuous.parameter_names
        )
        return rec

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
        # For batch size > 1, the acqf needs to support batching
        if batch_size > 1 and not self.acquisition_function.supports_batching:
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
                # If using CPU, first move everything to CPU
                if (
                    hasattr(self._botorch_acqf, "model")
                    and self._botorch_acqf.model is not None
                ):
                    self._botorch_acqf.model = self._botorch_acqf.model.to("cpu")
                    if hasattr(self._botorch_acqf.model, "train_inputs"):
                        self._botorch_acqf.model.train_inputs = tuple(
                            x.to("cpu") for x in self._botorch_acqf.model.train_inputs
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
        bounds_tensor = self._to_device(
            torch.from_numpy(searchspace.comp_rep_bounds.values)
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

        # Move points to CPU and detach before converting to numpy
        if points.is_cuda:
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

    def _setup_botorch_acqf(
        self,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame | None,
        pending_experiments: pd.DataFrame | None,
    ) -> None:
        """Set up the BoTorch acquisition function."""
        with single_device_mode(state=True):  # Force everything to same device
            # Important: Clear CUDA cache before setup to avoid memory conflicts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Call parent setup
            super()._setup_botorch_acqf(
                searchspace, objective, measurements, pending_experiments
            )

            # Move everything to the device and handle potential issues
            if hasattr(self, "device") and self.device is not None:
                # Force the acquisition function to the correct device if possible
                if hasattr(self, "_botorch_acqf") and self._botorch_acqf is not None:
                    if hasattr(self._botorch_acqf, "to"):
                        self._botorch_acqf = self._botorch_acqf.to(self.device)

                # Don't try to move the surrogate model directly since it might
                # not support it. Instead, we'll focus on the botorch_model if
                #  it exists
                if (
                    hasattr(self, "_botorch_model")
                    and self._botorch_model is not None
                    and hasattr(self._botorch_model, "to")
                ):
                    self._botorch_model = self._botorch_model.to(self.device)

                # Force garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
