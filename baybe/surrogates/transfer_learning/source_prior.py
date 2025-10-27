"""Abstract base class for source prior surrogates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from attrs import define, field
from botorch.models.model import Model
from botorch.posteriors import Posterior
from torch import Tensor
from typing_extensions import override

from baybe.exceptions import ModelNotTrainedError
from baybe.parameters import TaskParameter

if TYPE_CHECKING:
    from baybe.kernels.base import Kernel

from copy import deepcopy

import gpytorch
import torch
from botorch.models import SingleTaskGP
from botorch.posteriors import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal

from baybe.searchspace import SearchSpace
from baybe.surrogates import GaussianProcessSurrogate
from baybe.surrogates.gaussian_process.core import _ModelContext
from baybe.surrogates.gaussian_process.presets.default import (
    _default_noise_factory,
)


class GPBuilder:
    """Builds a GP from a pretrained prior GP, by using the posterior mean of the prior GP as the
    prior mean for the new GP.

    Args:
        searchspace: The search space (typically reduced, without TaskParameter).
        kernel_factory: The kernel factory from the surrogate (e.g., self.kernel_factory).
    """

    def __init__(
        self,
        searchspace: SearchSpace,
        kernel_factory: Kernel,
    ) -> None:
        """Initialize the GP builder.

        Args:
            searchspace: The search space (typically reduced, without TaskParameter).
            kernel_factory: The kernel factory from the surrogate.
        """
        self.searchspace: SearchSpace = searchspace
        self.kernel_factory: Kernel = kernel_factory
        self._context: _ModelContext = _ModelContext(searchspace)

    def create_gp(
        self,
        train_x: Tensor,
        train_y: Tensor,
        prior: SingleTaskGP | None = None,
        use_prior_mean: bool = False,
        use_prior_kernel: bool = False,
    ) -> SingleTaskGP:
        """Build a GP from a pretrained prior GP.

        If a prior is provided, its posterior mean is used as the mean function for the new GP.
        If no prior is provided, a standard GP is created. For the covariance, BayBE's kernel factory
        is used in both cases.

        Args:
            train_x: Training input data.
            train_y: Training target data.
            prior: Optional prior GP to use as mean function (for source_prior approach).
                  When provided, the prior GP's mean function will be used as the mean module
                  for the new GP, enabling transfer learning through prior knowledge.

        Returns:
            Fitted SingleTaskGP model using BayBE's kernel and noise configuration.

        Raises:
            RuntimeError: If GP fitting fails after multiple attempts.
        """
        import botorch
        import gpytorch

        if (use_prior_mean or use_prior_kernel) and prior is None:
            raise ValueError(
                "Prior GP must be provided when using prior mean or kernel."
            )
        if (use_prior_mean and use_prior_kernel):
            raise ValueError("Choose either mean or covariance transfer.")

        numerical_idxs = self._context.get_numerical_indices(train_x.shape[-1])

        # Configure mean module and transforms based on prior
        if prior is None or not use_prior_mean:
            # Standard GP without prior knowledge
            mean_module = gpytorch.means.ConstantMean()
            # For GPs, we let botorch handle the scaling.
            input_transform = botorch.models.transforms.Normalize(
                train_x.shape[-1],
                bounds=self._context.parameter_bounds,
                indices=numerical_idxs,
            )
            outcome_transform = botorch.models.transforms.Standardize(1)
        else:
            # Use the provided prior GP as mean function (source_prior approach)

            prior_model = deepcopy(prior)
            mean_module = GPyTMean(prior_model)
            input_transform = prior_model.input_transform
            outcome_transform = prior_model.outcome_transform

        # Extract the batch shape of the training data
        batch_shape = train_x.shape[:-2]
        mean_module.batch_shape = batch_shape

        # Define the covariance module for the numeric dimensions using BayBE's kernel factory
        #if prior is None or not use_prior_kernel:
        base_covar_module = self.kernel_factory(
            self._context.searchspace, train_x, train_y
        ).to_gpytorch(
            ard_num_dims=train_x.shape[-1] - self._context.n_task_dimensions,
            active_dims=numerical_idxs,
            batch_shape=batch_shape,
        )
        if use_prior_kernel:
            # Extract the source kernel and freeze its parameters
            #base_covar_module = GPyTKernel(prior.covar_module)
            prior_covar = prior.covar_module
            if hasattr(prior_covar,'kernels'):
                prior_base_kernel=prior_covar.kernels[0].base_kernel
            else:
                prior_base_kernel=prior_covar.base_kernel
            base_covar_module.base_kernel = GPyTKernel(kernel=deepcopy(prior_base_kernel))

            # For kernel transfer, also use source transforms for consistency
            input_transform = prior.input_transform
            outcome_transform = prior.outcome_transform
            #raise NotImplementedError("Using prior kernel is not yet implemented.")

        # # TODO: This can be removed.
        # # Handle multi-task kernels (though reduced searchspace shouldn't have tasks)
        # if not self._context.is_multitask:
        #     covar_module = base_covar_module
        # else:
        #     # This branch should rarely be hit since we work with reduced searchspace
        #     task_covar_module = gpytorch.kernels.IndexKernel(
        #         num_tasks=self._context.n_tasks,
        #         active_dims=self._context.task_idx,
        #         rank=self._context.n_tasks,
        #     )
        #     covar_module = base_covar_module * task_covar_module

        covar_module = base_covar_module

        # Create GP likelihood with BayBE's noise configuration
        noise_prior = _default_noise_factory(
            self._context.searchspace, train_x, train_y
        )
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=noise_prior[0].to_gpytorch(), batch_shape=batch_shape
        )
        likelihood.noise = torch.tensor([noise_prior[1]])

        # Construct and fit the Gaussian process
        model = botorch.models.SingleTaskGP(
            train_x,
            train_y,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            mean_module=mean_module,
            covar_module=covar_module,
            likelihood=likelihood,
        )

        # Fit the model using BayBE's standard MLL approach
        mll = gpytorch.mlls.LeaveOneOutPseudoLikelihood(model.likelihood, model)
        botorch.fit.fit_gpytorch_mll(mll, max_attempts=50)

        return model


class GPyTMean(gpytorch.means.Mean):
    """GPyTorch mean module using a trained GP as prior mean.

    This mean module wraps a trained Gaussian Process and uses its predictions
    as the mean function for another GP.
    """

    def __init__(
        self, gp: SingleTaskGP, batch_shape: torch.Size = torch.Size(), **kwargs: Any
    ) -> None:
        """Initialize the GP-based mean module.

        Args:
            gp: Trained Gaussian Process to use as mean function.
            batch_shape: Batch shape for the mean module.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        # See https://github.com/cornellius-gp/gpytorch/issues/743
        self.gp: SingleTaskGP = deepcopy(gp)
        self.batch_shape: torch.Size = batch_shape
        for param in self.gp.parameters():
            param.requires_grad = False

    def reset_gp(self) -> None:
        """Reset the GP to evaluation mode for prediction."""
        self.gp.eval()
        self.gp.likelihood.eval()

    def forward(self, input: Tensor) -> Tensor:
        """Compute the mean function using the wrapped GP.

        Args:
            input: Input tensor for which to compute the mean.

        Returns:
            Mean predictions from the wrapped GP.
        """
        self.reset_gp()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            with gpytorch.settings.detach_test_caches(False):
                mean = self.gp(input).mean.detach()
        mean = mean.reshape(torch.broadcast_shapes(self.batch_shape, input.shape[:-1]))
        return mean


class GPyTKernel(gpytorch.kernels.Kernel):
    """GPyTorch kernel module wrapping a pre-trained kernel.

    This kernel module wraps a trained kernel and uses it as a fixed kernel
    component in another GP. The wrapped kernel's parameters are frozen.
    """

    def __init__(self, kernel, **kwargs):
        """Initialize the kernel wrapper.

        Args:
            kernel: Pre-trained kernel to wrap.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        # See https://github.com/cornellius-gp/gpytorch/issues/743
        self.base_kernel = deepcopy(kernel)
        for param in self.base_kernel.parameters():
            param.requires_grad = False

    def reset(self):
        """Reset the wrapped kernel to evaluation mode."""
        self.base_kernel.eval()

    def forward(self, x1, x2, **params):
        """Compute kernel matrix using the wrapped kernel.

        Args:
            x1: First set of input points.
            x2: Second set of input points.
            **params: Additional kernel parameters.

        Returns:
            Kernel matrix computed by the wrapped kernel.
        """
        self.reset()
        # Allow gradient flow for input optimization while keeping kernel parameters frozen
        # The kernel parameters are already frozen via requires_grad=False in __init__
        with gpytorch.settings.fast_pred_var():
            with gpytorch.settings.detach_test_caches(False):
                k = self.base_kernel.forward(x1, x2, **params)
        return k


class SourcePriorWrapperModel(Model):
    """BoTorch Model wrapper for SourcePriorGaussianProcessSurrogate."""

    def __init__(
        self, source_prior_surrogate: SourcePriorGaussianProcessSurrogate
    ) -> None:
        """Initialize the wrapper model.

        Args:
            source_prior_surrogate: The SourcePriorGaussianProcessSurrogate to wrap.
        """
        super().__init__()
        self.surrogate: SourcePriorGaussianProcessSurrogate = source_prior_surrogate

    @property
    def num_outputs(self) -> int:
        """Number of outputs of the model."""
        return 1

    def posterior(self, X: Tensor, **kwargs: Any) -> Posterior:
        """Compute posterior distribution.

        Args:
            X: Input tensor with shape (..., n_points, n_features + 1)
               where the last feature is the task index.
            **kwargs: Additional keyword arguments.

        Returns:
            Posterior distribution over the input points.
        """
        return self.surrogate._posterior(X)

    @property
    def train_inputs(self) -> tuple[Tensor, ...]:
        """Return training inputs from the target GP.

        Returns:
            Tuple of training input tensors.

        Raises:
            RuntimeError: If model not fitted.
        """
        if self.surrogate._target_gp is None:
            raise RuntimeError("Model not fitted")
        return self.surrogate._target_gp.train_inputs

    @property
    def train_targets(self) -> Tensor:
        """Return training targets from the target GP.

        Returns:
            Training target tensor.

        Raises:
            RuntimeError: If model not fitted.
        """
        if self.surrogate._target_gp is None:
            raise RuntimeError("Model not fitted")
        return self.surrogate._target_gp.train_targets


@define
class SourcePriorGaussianProcessSurrogate(GaussianProcessSurrogate):
    """Source prior transfer learning Gaussian process surrogate.

    This surrogate implements transfer learning by:
    1. Training a source GP on source task data (without task dimension)
    2. Using the source GP as a mean prior for the target GP
    3. Training the target GP on all data (source + target) with source-informed priors
    """

    # Class variables
    supports_transfer_learning: ClassVar[bool] = True
    """Class variable encoding transfer learning support."""

    supports_multi_output: ClassVar[bool] = False
    """Class variable encoding multi-output compatibility."""

    # Input dim of the problem
    input_dim: int | None = field(default=None)
    """Dimensionality of the input space (excluding task feature)."""

    use_prior_mean: bool = field(default=True)
    """Whether to use the source GP as prior mean for the target GP."""

    use_prior_kernel: bool = field(default=False)
    """Whether to use the source GP's kernel as prior kernel for the target GP."""

    numerical_stability: bool = field(default=True)
    """Whether to use numerically stable implementation."""

    # Private attributes for storing model state
    _model: Model | None = field(init=False, default=None, eq=False)
    """The actual SourcePriorModel instance."""

    _target_task_id: int | None = field(init=False, default=None, eq=False)
    """Numeric ID of the target task."""

    _task_column_idx: int | None = field(init=False, default=None, eq=False)
    """Column index of task feature in BayBE's computational representation."""

    _source_gp: SingleTaskGP | None = field(init=False, default=None, eq=False)
    """Fitted source Gaussian Process model."""

    _target_gp: SingleTaskGP | None = field(init=False, default=None, eq=False)
    """Fitted target Gaussian Process model with source prior."""

    def _identify_target_task(self) -> tuple[int, float]:
        """Identify the TaskParameter and return its column index and target value.

        This function identifies the TaskParameter within the search space, retrieves
        its active value, and returns both the column index of the TaskParameter and
        the computational representation value for the active value. This is useful
        for filtering tensor rows that correspond to the target task.

        Returns:
            A tuple containing:
            - task_idx (int): The column index of the TaskParameter in the computational
                            representation of the search space
            - target_value (float): The computational representation value for the
                        active value of the TaskParameter (used for filtering)
        """
        searchspace = self._searchspace  # type: ignore[assignment]
        # Find the TaskParameter in the search space
        task_params = [
            p for p in searchspace.parameters if isinstance(p, TaskParameter)
        ]

        task_param = task_params[0]

        # Get the active value
        active_value = task_param.active_values[0]

        # Get the index of the TaskParameter in the computational representation
        task_idx = searchspace.task_idx  # type: ignore[assignment]

        # Get the computational representation value for the active value
        # TaskParameter uses INT encoding, so comp_df has a single column with
        # integer values
        comp_df = task_param.comp_df

        # Extract the single computational representation value
        target_value = float(comp_df.loc[active_value].iloc[0])

        return task_idx, target_value

    def _validate_transfer_learning_context(self) -> None:
        """Validate that we have a proper transfer learning setup.

        Raises:
            ValueError: If no task parameter found in search space.
            ValueError: If input dimensions don't match expected format.
        """
        if self._searchspace.task_idx is None:
            raise ValueError(
                "No task parameter found in search space. "
                "SourcePriorGaussianProcessSurrogate requires a TaskParameter "
                "for transfer learning."
            )
        # Set input_dim if not provided at initialization
        if self.input_dim is None:
            self.input_dim = len(self._searchspace.comp_rep_columns) - 1

        # Validate that we have the expected number of feature dimensions
        expected_total_dims = self.input_dim + 1  # features + task
        actual_total_dims = len(self._searchspace.comp_rep_columns)

        if actual_total_dims != expected_total_dims:
            raise ValueError(
                f"Expected {expected_total_dims} total dimensions "
                f"({self.input_dim} features + 1 task), "
                f"but got {actual_total_dims} from search space."
            )

    def _extract_task_data(
        self,
        X: Tensor,
        Y: Tensor | None = None,
        task_feature: int = -1,
        target_task: int = 0,
    ) -> tuple[list[tuple[Tensor, Tensor | None]], tuple[Tensor, Tensor | None]]:
        """Extract source and target data from multi-task format.

        Args:
            X: Input data including task indices, shape (n_total, input_dim + 1).
            Y: Output data, shape (n_total, 1).
            task_feature: Index of the task feature dimension.
            target_task: Task ID for the target task.

        Returns:
            Tuple of (source_data_list, target_data_tuple).
        """
        # Extract task indices
        task_indices = X[:, task_feature].long()

        # Extract input features (without task index)
        if task_feature == -1:
            X_features = X[:, :-1]
        else:
            X_features = torch.cat(
                [X[:, :task_feature], X[:, task_feature + 1 :]], dim=1
            )

        # Get unique task IDs and sort them
        unique_tasks = torch.unique(task_indices)
        source_tasks = unique_tasks[unique_tasks != target_task].sort().values

        # Extract source data
        source_data = []
        for task_id in source_tasks:
            mask = task_indices == task_id
            X_task = X_features[mask]
            Y_task = Y[mask] if Y is not None else None
            source_data.append((X_task, Y_task))

        # Extract target data
        target_mask = task_indices == target_task
        X_target = X_features[target_mask]
        Y_target = Y[target_mask] if Y is not None else None

        return source_data, (X_target, Y_target)

    def _reduce_searchspace(self, searchspace: SearchSpace) -> SearchSpace:
        """Remove TaskParameter from a SearchSpace if it exists.

        Args:
            searchspace: The SearchSpace to process.

        Returns:
            A new SearchSpace without TaskParameter, or the original SearchSpace
            if no TaskParameter exists.
        """
        # Get all parameters from the search space
        parameters = list(searchspace.parameters)

        # Filter out TaskParameter instances
        filtered_parameters = [
            param for param in parameters if not isinstance(param, TaskParameter)
        ]

        # If no TaskParameter was found, return the original searchspace
        if len(filtered_parameters) == len(parameters):
            return searchspace

        # If all parameters were TaskParameters, create empty SearchSpace
        if not filtered_parameters:
            return SearchSpace()

        # Create new SearchSpace with filtered parameters and constraints
        return SearchSpace.from_product(
            parameters=filtered_parameters,
            constraints=list(searchspace.constraints),
        )

    @override
    def _fit(self, train_x: Tensor, train_y: Tensor) -> None:
        """Fit the transfer learning model.

        This method handles the common training workflow for all transfergpbo models:
        1. Validate transfer learning context
        2. Identify target task from TaskParameter
        3. Fit source GP on the source task data
        4. Fit target GP on the target task data using source GP posterior mean
           as prior mean.

        Args:
            train_x: Training inputs in BayBE's computational representation.
                    Shape: (n_points, n_features + 1) last column may be task indices.
            train_y: Training targets. Shape: (n_points, 1).

        Raises:
            ValueError: If received empty training data or if context is invalid.
        """
        # FIXME[typing]: It seems there is currently no better way to inform the type
        #   checker that the attribute is available at the time of the function call
        assert self._searchspace is not None

        # Check if we receive empty data
        if train_x.shape[0] == 0 or train_y.shape[0] == 0:
            raise ValueError(
                f"Received empty training data! "
                f"train_x.shape={train_x.shape}, train_y.shape={train_y.shape}"
            )

        # 1. Validate transfer learning context
        self._validate_transfer_learning_context()

        # 2. Identify target task from TaskParameter active_values
        self._task_column_idx, self._target_task_id = self._identify_target_task()

        source_data, (X_target, Y_target) = self._extract_task_data(
            train_x, train_y, self._task_column_idx, self._target_task_id
        )

        if len(source_data) == 0:
            raise ValueError(
                "No source data found. SourcePriorGaussianProcessSurrogate requires "
                "at least source data for transfer learning."
            )

        source_data = source_data[0]
        X_source, Y_source = source_data

        # Remove task parameter from searchspace before training the GPs
        reduced_searchspace = self._reduce_searchspace(searchspace=self._searchspace)

        # Create shared GP builder for consistent GP construction
        gp_builder = GPBuilder(
            searchspace=reduced_searchspace, kernel_factory=self.kernel_factory
        )

        # Fit the source GP using shared builder
        self._source_gp = gp_builder.create_gp(
            train_x=X_source, train_y=Y_source, prior=None
        )

        # Handle target data
        if X_target.shape[0] == 0:
            # No target data available - use a copy of source GP as target GP
            print(
                "No target data provided for SourcePrior model. Using copy of source GP as target GP."
            )
            from copy import deepcopy

            self._target_gp = deepcopy(self._source_gp)
        else:
            # Target data available - fit target model using source posterior as prior
            self._target_gp = gp_builder.create_gp(
                train_x=X_target,
                train_y=Y_target,
                prior=self._source_gp,
                use_prior_mean=self.use_prior_mean,
                use_prior_kernel=self.use_prior_kernel,
            )

    @override
    def _posterior(self, candidates_comp_scaled: Tensor, /) -> Posterior:
        """Compute posterior predictions.

        This method handles prediction for candidates that may contain
        task indices for any of the available tasks (source or target).

        Args:
            candidates_comp_scaled: Candidate points in computational
                representation. Should include task indices in same format as
                training data.

        Returns:
            Posterior distribution for the candidate points.

        Raises:
            ModelNotTrainedError: If model hasn't been trained yet.
        """
        if self._target_gp is None:
            raise ModelNotTrainedError(
                "Model must be fitted before making predictions. Call fit() first."
            )

        # Handle non-batched inputs by adding a batch dimension
        if candidates_comp_scaled.dim() == 2:
            candidates_comp_scaled = candidates_comp_scaled.unsqueeze(0)
            unbatch_output = True
        else:
            unbatch_output = False

        # Save original shape for reshaping outputs later
        batch_shape = candidates_comp_scaled.shape[
            :-2
        ]  # Everything except the last two dimensions
        n_points = candidates_comp_scaled.shape[-2]  # Number of points

        # Get dtype from the target GP to ensure consistency
        gp_dtype = next(self._target_gp.parameters()).dtype
        device = candidates_comp_scaled.device

        # Process each batch element separately
        batch_means = []
        batch_covs = []

        # Iterate over batch elements
        for batch_idx in range(batch_shape.numel()):
            # Get flat batch index
            batch_indices = []
            remaining = batch_idx
            for dim_size in reversed(batch_shape):
                batch_indices.insert(0, remaining % dim_size)
                remaining = remaining // dim_size

            # Extract data for this batch
            if batch_shape.numel() == 1:
                # Single batch dimension
                candidates_batch = candidates_comp_scaled[batch_idx]
            else:
                # Multiple batch dimensions
                candidates_batch = candidates_comp_scaled[tuple(batch_indices)]

            # Initialize output tensors for this batch
            batch_mean = torch.zeros(n_points, 1, dtype=gp_dtype, device=device)
            batch_cov = torch.eye(n_points, dtype=gp_dtype, device=device) * 1e-6

            # Extract task indices from candidates
            task_indices = candidates_batch[:, self._task_column_idx].long()
            target_mask = task_indices == self._target_task_id
            source_mask = ~target_mask

            # Extract source and target candidate points
            source_data, (X_target, Y_target) = self._extract_task_data(
                X=candidates_batch,
                Y=None,
                task_feature=self._task_column_idx,
                target_task=self._target_task_id,
            )

            # Handle target predictions if we have target candidates
            if target_mask.any():
                target_posterior = self._target_gp.posterior(X_target)
                target_mean = target_posterior.mean
                target_var = target_posterior.variance

                # Fill in target predictions
                batch_mean[target_mask] = target_mean

                # Fill target covariance (diagonal approximation for simplicity)
                target_indices = torch.where(target_mask)[0]
                for i, idx in enumerate(target_indices):
                    batch_cov[idx, idx] = target_var[i]

            # Handle source predictions if we have source candidates
            if source_mask.any() and len(source_data) > 0:
                X_source, Y_source = source_data[0]

                # Use source GP for source predictions
                source_posterior = self._source_gp.posterior(X_source)
                source_mean = source_posterior.mean
                source_var = source_posterior.variance

                # Fill in source predictions
                batch_mean[source_mask] = source_mean

                # Fill source covariance (diagonal approximation for simplicity)
                source_indices = torch.where(source_mask)[0]
                for i, idx in enumerate(source_indices):
                    batch_cov[idx, idx] = source_var[i]

            batch_means.append(batch_mean)
            batch_covs.append(batch_cov)

        # Stack results along batch dimension
        if batch_shape.numel() == 1:
            # Single batch dimension
            stacked_means = torch.stack(batch_means, dim=0)
            stacked_covs = torch.stack(batch_covs, dim=0)
        else:
            # Multiple batch dimensions - reshape to original batch shape
            stacked_means = torch.stack(batch_means, dim=0).reshape(
                *batch_shape, n_points, 1
            )
            stacked_covs = torch.stack(batch_covs, dim=0).reshape(
                *batch_shape, n_points, n_points
            )

        # Remove extra batch dimension if input wasn't batched
        if unbatch_output:
            stacked_means = stacked_means.squeeze(0)
            stacked_covs = stacked_covs.squeeze(0)

        # Create the MultivariateNormal distribution
        mvn = MultivariateNormal(
            stacked_means.squeeze(-1),  # Remove last dimension: (..., n, 1) -> (..., n)
            stacked_covs,
        )
        posterior = GPyTorchPosterior(mvn)

        return posterior

    @override
    def to_botorch(self) -> Model:
        """Return the BoTorch model representation.

        Returns:
            A wrapper model that can handle both source and target predictions.

        Raises:
            ModelNotTrainedError: If model hasn't been trained yet.
        """
        if self._target_gp is None:
            raise ModelNotTrainedError(
                "Model must be fitted before accessing BoTorch model."
            )
        return SourcePriorWrapperModel(self)

