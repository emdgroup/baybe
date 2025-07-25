"""Torch Models for "Transfer Learning with GPs for BO" by Tighineanu et al. (2022)."""

from typing import List, Optional, Tuple

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

    Note:
        This is the basic implementation without numerical stability enhancements.
        For production use with small datasets or ill-conditioned problems,
        consider using :class:`MHGPModelStable` instead.

    Examples:
        >>> # Create model and fit with MultiTaskGP-like interface
        >>> model = MHGPModel(input_dim=2)
        >>> # X_multi includes task indices, Y contains all outputs
        >>> model.meta_fit(X_multi, Y, task_feature=-1, target_task=2)
        >>> model.fit(X_multi, Y, task_feature=-1, target_task=2)
        >>>
        >>> # Make predictions
        >>> posterior = model.posterior(X_test)
        >>> mean = posterior.mean
        >>> variance = posterior.variance
    """

    def __init__(self, input_dim: int) -> None:
        """Initialize the MHGP model.

        Args:
            input_dim: Dimensionality of the input space (excluding task feature).
        """
        super().__init__()
        self.input_dim = input_dim

        self.source_gps: List[SingleTaskGP] = []
        """List of fitted source Gaussian Process models."""

        self.target_gp: Optional[SingleTaskGP] = None
        """The target Gaussian Process model."""

        self._fitted: bool = False
        """Whether the model has been fully fitted (including target task)."""

    @property
    def num_outputs(self) -> int:
        """Number of outputs of the model."""
        return 1

    def _extract_task_data(
        self, X: Tensor, Y: Tensor, task_feature: int, target_task: int
    ) -> Tuple[List[Tuple[Tensor, Tensor]], Tuple[Tensor, Tensor]]:
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

        Raises:
            RuntimeError: If fitting any of the source GPs fails.
        """
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

        Raises:
            RuntimeError: If fitting the target GP fails.
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

        # Extract task indices (assume last column contains task indices)
        task_indices = X[:, -1].long()
        X_features = X[:, :-1]  # Remove task indices

        # Get unique tasks and validate
        unique_tasks = torch.unique(task_indices)
        max_task_id = len(self.source_gps)  # Target task has ID = len(source_gps)

        # Validate task indices
        if torch.any(task_indices < 0) or torch.any(task_indices > max_task_id):
            raise ValueError(
                f"Task indices must be in range [0, {max_task_id}]. "
                f"Got task indices: {unique_tasks.tolist()}"
            )

        with torch.no_grad():
            # Initialize output tensors
            n_points = X_features.shape[0]
            final_means = torch.zeros(n_points, 1, dtype=X.dtype, device=X.device)
            final_cov = torch.zeros(n_points, n_points, dtype=X.dtype, device=X.device)

            # Process each unique task
            for task_id in unique_tasks:
                task_mask = task_indices == task_id
                X_task = X_features[task_mask]

                # Prediction from source GP
                task_mean, task_var = self._predict_from_stack(X_task, task_id.item())

                # Store results
                final_means[task_mask] = task_mean
                final_cov[task_mask] = task_var

            return GPyTorchPosterior(
                MultivariateNormal(final_means.squeeze(-1), final_cov)
            )

    def _predict_from_stack(
        self, X: Tensor, up_to_task_id: int
    ) -> Tuple[Tensor, Tensor]:
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
        >>> # Create stable model with MultiTaskGP-like interface
        >>> model = MHGPModelStable(input_dim=2)
        >>> # X_multi includes task indices, Y contains all outputs
        >>> model.meta_fit(X_multi, Y, task_feature=-1, target_task=2)
        >>> model.fit(X_multi, Y, task_feature=-1, target_task=2)
        >>>
        >>> # Predictions are computed the same way
        >>> posterior = model.posterior(X_test)
    """

    @override
    def _predict_from_stack(
        self, X: Tensor, up_to_task_id: int
    ) -> Tuple[Tensor, Tensor]:
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
