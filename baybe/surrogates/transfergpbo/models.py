"""Torch Models for "Transfer Learning with GPs for BO" by Tighineanu et al. (2022)."""

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.posteriors import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor
from typing_extensions import override

from baybe.surrogates.transfergpbo.utils import is_pd, nearest_pd


class MHGPModel(Model):
    """Multi-task Hierarchical Gaussian Process model for transfer learning.

    This model implements the MHGP (Multi-task Hierarchical GP) approach from
    "Transfer Learning with Gaussian Processes for Bayesian Optimization" by
    Tighineanu et al. (2022). The model sequentially trains a stack of Gaussian
    processes where each GP models the residuals from the previous GPs in the stack.

    The key idea is that each GP uses the posterior mean of the previous GP as its
    prior mean, creating a hierarchical structure that enables transfer learning
    from source tasks to a target task.

    Args:
        input_dim: Dimensionality of the input space (excluding task feature).

    Note:
        This is the basic implementation without numerical stability enhancements.
        For production use with small datasets or ill-conditioned problems,
        consider using :class:`MHGPModelStable` instead.

    Examples:
        >>> import torch
        >>>
        >>> # Create model and fit with MultiTaskGP-like interface
        >>> model = MHGPModel(input_dim=2)
        >>>
        >>> # X_multi includes task indices, Y contains all outputs
        >>> X_multi = torch.tensor([[0.1, 0.2, 0], [0.3, 0.4, 0], [0.5, 0.6, 1]])
        >>> Y = torch.tensor([[0.5], [0.7], [0.9]])
        >>>
        >>> model.meta_fit(X_multi, Y, task_feature=-1, target_task=1)
        >>> model.fit(X_multi, Y, task_feature=-1, target_task=1)
        >>>
        >>> # Make predictions
        >>> X_test = torch.tensor([[0.1, 0.4, 0], [0.5, 0.8, 1], [0.7, 0.8, 1]])
        >>> posterior = model.posterior(X_test)
        >>> mean = posterior.mean
        >>> variance = posterior.variance
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        """Input dimension excluding TaskParameter"""
        self.task_feature: int | None = None
        """The index of the task descriptors in the data"""
        self.target_task: int | None = None
        """The descriptor of the target task"""
        self.source_gps: list[SingleTaskGP] = []
        """List of fitted source Gaussian Process models."""
        self.target_gp: SingleTaskGP | None = None
        """The target Gaussian Process model."""
        self._fitted: bool = False
        """Whether the model has been fully fitted (including target task)."""

    @property
    def num_outputs(self) -> int:
        """Number of outputs of the model."""
        return 1

    def _extract_task_data(
        self, X: Tensor, Y: Tensor, task_feature: int, target_task: int
    ) -> tuple[list[tuple[Tensor, Tensor]], tuple[Tensor, Tensor]]:
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
            Y_task = Y[mask]
            source_data.append((X_task, Y_task))

        # Extract target data
        target_mask = task_indices == target_task
        X_target = X_features[target_mask]
        Y_target = Y[target_mask]

        return source_data, (X_target, Y_target)

    def meta_fit(
        self, X: Tensor, Y: Tensor, task_feature: int = -1, target_task: int = 2
    ) -> None:
        """Fit source GPs sequentially on residuals.

        This method implements the core MHGP training procedure by fitting each
        source GP to the residuals left by the previous GPs in the stack.

        Args:
            X: Input data including task indices, shape (n_total, input_dim + 1).
            Y: Output data, shape (n_total, 1).
            task_feature: Index of the task feature dimension (default: -1).
            target_task: Task ID for the target task (default: 2).
        """
        # Store task configuration
        self.task_feature = task_feature
        self.target_task = target_task
        # Extract source and target data
        source_data, _ = self._extract_task_data(X, Y, task_feature, target_task)

        for i, (X_source, Y_source) in enumerate(source_data):
            # Compute residuals from previous GPs
            if i == 0:
                residuals = Y_source.clone()
            else:
                residuals = Y_source.clone()
                for j in range(i):
                    with torch.no_grad():
                        pred_mean = self.source_gps[j].posterior(X_source).mean
                        residuals = residuals - pred_mean.detach()

            # Ensure residuals are properly detached and cloned
            residuals = residuals.detach().clone()
            X_source_clean = X_source.detach().clone()

            # Create and fit new GP
            gp = SingleTaskGP(X_source_clean, residuals)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

            self.source_gps.append(gp)

    def fit(
        self, X: Tensor, Y: Tensor, task_feature: int = -1, target_task: int = 2
    ) -> None:
        """Fit target GP on residuals from all source GPs.

        This method completes the MHGP training by fitting the final target GP
        to the residuals left after removing predictions from all source GPs.

        Args:
            X: Input data including task indices, shape (n_total, input_dim + 1).
            Y: Output data, shape (n_total, 1).
            task_feature: Index of the task feature dimension (default: -1).
            target_task: Task ID for the target task (default: 2).
        """
        # Extract target data
        _, (X_target, Y_target) = self._extract_task_data(
            X, Y, task_feature, target_task
        )

        if len(self.source_gps) == 0:
            residuals = Y_target.clone()
        else:
            residuals = Y_target.clone()
            for gp in self.source_gps:
                with torch.no_grad():
                    pred_mean = gp.posterior(X_target).mean
                    residuals = residuals - pred_mean.detach()

        # Ensure clean tensors
        residuals = residuals.detach().clone()
        X_target_clean = X_target.detach().clone()

        # Create and fit target GP
        self.target_gp = SingleTaskGP(X_target_clean, residuals)
        mll = ExactMarginalLogLikelihood(self.target_gp.likelihood, self.target_gp)
        fit_gpytorch_mll(mll)

        self._fitted = True

    @override
    def posterior(self, X: Tensor, **kwargs) -> GPyTorchPosterior:
        """Compute posterior distribution with multi-task interface.

        This method now accepts input data with task indices (like MultiTaskGP) and
        returns predictions from the corresponding model in the stack for each input.

        Args:
            X: Input locations including task indices, with shape
                (batch_shape, n_points, input_dim + 1). The last column should
                contain task indices.
            **kwargs: Additional keyword arguments (currently unused).

        Returns:
            A GPyTorchPosterior object representing the posterior distribution
            at the input locations from the corresponding models in the stack.

        Raises:
            RuntimeError: If the model has not been fitted yet.
            ValueError: If task indices are invalid or out of range.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted first. Call meta_fit() and fit().")
        if self.task_feature is None:
            raise RuntimeError("Task feature not configured. Call meta_fit() first.")

        # Handle non-batched inputs by adding a batch dimension
        if X.dim() == 2:
            X = X.unsqueeze(0)
            unbatch_output = True
        else:
            unbatch_output = False

        # Save original shape for reshaping outputs later
        batch_shape = X.shape[:-2]  # Everything except the last two dimensions
        n_points = X.shape[-2]  # Number of points

        # Extract task feature index
        task_feature = self.task_feature

        # Extract task indices and features
        if task_feature == -1:
            # Task is last column
            task_indices = X[..., -1].long()
            X_features = X[..., :-1]
        else:
            # Task is at specified position
            task_indices = X[..., task_feature].long()
            # Remove task column while preserving batch dimensions
            if task_feature == 0:
                # Task is first column
                X_features = X[..., 1:]
            elif task_feature == X.shape[-1] - 1:
                # Task is last column
                X_features = X[..., :-1]
            else:
                # Task is in the middle
                X_features = torch.cat(
                    [X[..., :task_feature], X[..., task_feature + 1 :]], dim=-1
                )

        # Validate max task ID
        max_task_id = len(self.source_gps)
        if torch.any(task_indices < 0) or torch.any(task_indices > max_task_id):
            raise ValueError(
                f"Task indices must be in range [0, {max_task_id}]. "
                f"Got invalid task indices."
            )

        # Process each batch element separately
        batch_means = []
        batch_covs = []

        with torch.set_grad_enabled(X.requires_grad):
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
                    task_indices_batch = task_indices[batch_idx]
                    X_features_batch = X_features[batch_idx]
                else:
                    # Multiple batch dimensions
                    task_indices_batch = task_indices[tuple(batch_indices)]
                    X_features_batch = X_features[tuple(batch_indices)]

                # Initialize output tensors for this batch
                batch_mean = torch.zeros(n_points, 1, dtype=X.dtype, device=X.device)
                batch_cov = torch.zeros(
                    n_points, n_points, dtype=X.dtype, device=X.device
                )

                # Process each unique task in this batch
                unique_tasks_batch = torch.unique(task_indices_batch)
                for task_id in unique_tasks_batch:
                    task_mask = task_indices_batch == task_id
                    task_indices_list = torch.where(task_mask)[0]
                    X_task = X_features_batch[task_mask]

                    # Prediction from source GP
                    task_mean, task_cov = self._predict_from_stack(
                        X_task, task_id.item()
                    )

                    # Store results
                    batch_mean[task_mask] = task_mean

                    # Assign covariance block using meshgrid indexing
                    idx_i, idx_j = torch.meshgrid(
                        task_indices_list, task_indices_list, indexing="ij"
                    )
                    batch_cov[idx_i, idx_j] = task_cov

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

        return GPyTorchPosterior(mvn)

    def _predict_from_stack(
        self, X: Tensor, up_to_task_id: int
    ) -> tuple[Tensor, Tensor]:
        """Predict using the stack up to the specified task ID.

        Args:
            X: Input features (without task indices).
            up_to_task_id: Task ID to predict up to (inclusive).

        Returns:
            Tuple of (mean, variance) predictions.
        """
        # Sum predictions from source GPs up to the specified task
        total_mean = torch.zeros(X.shape[0], 1, dtype=X.dtype, device=X.device)

        # Check if we need to include target GP
        if up_to_task_id == len(self.source_gps):
            for gp in self.source_gps:
                gp_posterior = gp.posterior(X)
                total_mean += gp_posterior.mean

            # Add target GP prediction
            target_posterior = self.target_gp.posterior(X)
            total_mean = total_mean + target_posterior.mean

            # Apply numerical stability fixes (TransferGPBO style)
            covar_matrix = target_posterior.covariance_matrix

        else:
            for task_id in range(up_to_task_id):
                gp_posterior = self.source_gps[task_id].posterior(X)
                total_mean += gp_posterior.mean

            # Apply numerical stability fixes (TransferGPBO style)
            covar_matrix = self.source_gps[up_to_task_id].posterior(X).covariance_matrix

        return total_mean, covar_matrix


class MHGPModelStable(MHGPModel):
    """Numerically stable Multi-task Hierarchical Gaussian Process model.

    This is an enhanced version of :class:`MHGPModel` that includes numerical
    stability improvements to handle small datasets, ill-conditioned covariance
    matrices, and other numerical issues that can arise in practice.

    The stability enhancements include:
    - Nearest positive-definite matrix computation for covariance matrices
    - Positive definiteness checks during prediction

    This version is recommended for production use, especially with small datasets
    or when numerical stability is a concern.

    Examples:
        >>> import torch
        >>>
        >>> # Create model and fit with MultiTaskGP-like interface
        >>> model = MHGPModelStable(input_dim=2)
        >>>
        >>> # X_multi includes task indices, Y contains all outputs
        >>> X_multi = torch.tensor([[0.1, 0.2, 0], [0.3, 0.4, 0], [0.5, 0.6, 1]])
        >>> Y = torch.tensor([[0.5], [0.7], [0.9]])
        >>>
        >>> model.meta_fit(X_multi, Y, task_feature=-1, target_task=1)
        >>> model.fit(X_multi, Y, task_feature=-1, target_task=1)
        >>>
        >>> # Make predictions
        >>> X_test = torch.tensor([[0.1, 0.4, 0], [0.5, 0.8, 1], [0.7, 0.8, 1]])
        >>> posterior = model.posterior(X_test)
        >>> mean = posterior.mean
        >>> variance = posterior.variance
    """

    @override
    def _predict_from_stack(
        self, X: Tensor, up_to_task_id: int
    ) -> tuple[Tensor, Tensor]:
        """Predict using the stack up to the specified task ID.

        Args:
            X: Input features (without task indices).
            up_to_task_id: Task ID to predict up to (inclusive).

        Returns:
            Tuple of (mean, variance) predictions.
        """
        # Sum predictions from source GPs up to the specified task
        total_mean = torch.zeros(X.shape[0], 1, dtype=X.dtype, device=X.device)

        # Check if we need to include target GP
        if up_to_task_id == len(self.source_gps):
            for gp in self.source_gps:
                gp_posterior = gp.posterior(X)
                total_mean += gp_posterior.mean

            # Add target GP prediction
            target_posterior = self.target_gp.posterior(X)
            total_mean = total_mean + target_posterior.mean

            # Apply numerical stability fixes (TransferGPBO style)
            covar_matrix = target_posterior.covariance_matrix

        else:
            for task_id in range(up_to_task_id):
                gp_posterior = self.source_gps[task_id].posterior(X)
                total_mean += gp_posterior.mean

            # Apply numerical stability fixes (TransferGPBO style)
            covar_matrix = self.source_gps[up_to_task_id].posterior(X).covariance_matrix

        # Check if covariance is positive definite, fix if not
        if not is_pd(covar_matrix):
            covar_matrix = nearest_pd(covar_matrix)

        return total_mean, covar_matrix
