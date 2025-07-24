"""Torch Models for "Transfer Learning with GPs for BO" by Tighineanu et al. (2022)."""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import MultiTaskGP, SingleTaskGP
from botorch.models.model import Model
from botorch.posteriors import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor
from typing_extensions import override

from baybe.surrogates.transfergpbo.utils import is_pd, nearest_pd


class MHGPModelStable(Model):
    """Numerically stable Multi-task Hierarchical GP following TransferGPBO approach.

    This implementation includes minimal numerical stability fixes based on the
    TransferGPBO codebase: iterative Cholesky regularization and nearest PD matrix
    computation for covariance matrices.
    """

    def __init__(self, input_dim: int) -> None:
        """Initialize the stable MHGP model.

        Args:
            input_dim: Dimensionality of the input space.
        """
        super().__init__()
        self.input_dim = input_dim

        self.source_gps: List[SingleTaskGP] = []
        """List of fitted source Gaussian Process models."""

        self.target_gp: Optional[SingleTaskGP] = None
        """The target Gaussian Process model."""

        self._fitted: bool = False
        """Whether the model has been fully fitted."""

    @property
    def num_outputs(self) -> int:
        """Number of outputs of the model."""
        return 1

    def meta_fit(self, source_data: List[Tuple[Tensor, Tensor]]) -> None:
        """Fit source GPs sequentially on residuals.

        Args:
            source_data: List of (X, Y) tuples for each source task.
        """
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

            # Clean tensors
            residuals = residuals.detach().clone()
            X_source_clean = X_source.detach().clone()

            # Create and fit GP
            gp = SingleTaskGP(X_source_clean, residuals)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

            self.source_gps.append(gp)

    def fit_target(self, X_target: Tensor, Y_target: Tensor) -> None:
        """Fit target GP on residuals from all source GPs.

        Args:
            X_target: Target task input data.
            Y_target: Target task output data.
        """
        if len(self.source_gps) == 0:
            residuals = Y_target.clone()
        else:
            residuals = Y_target.clone()
            for gp in self.source_gps:
                with torch.no_grad():
                    pred_mean = gp.posterior(X_target).mean
                    residuals = residuals - pred_mean.detach()

        # Clean tensors
        residuals = residuals.detach().clone()
        X_target_clean = X_target.detach().clone()

        # Create and fit target GP
        self.target_gp = SingleTaskGP(X_target_clean, residuals)
        mll = ExactMarginalLogLikelihood(self.target_gp.likelihood, self.target_gp)
        fit_gpytorch_mll(mll)

        self._fitted = True

    @override
    def posterior(self, X: Tensor) -> GPyTorchPosterior:
        """Compute posterior with numerical stability fixes.

        Args:
            X: Input locations for prediction.

        Returns:
            Posterior distribution with numerical stability fixes applied.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted first.")

        with torch.no_grad():
            # Sum predictions from all source GPs
            total_mean = torch.zeros(X.shape[0], 1, dtype=X.dtype, device=X.device)

            for gp in self.source_gps:
                gp_posterior = gp.posterior(X)
                total_mean += gp_posterior.mean

            # Add target GP prediction
            target_posterior = self.target_gp.posterior(X)
            final_mean = total_mean + target_posterior.mean

            # Apply numerical stability fixes (TransferGPBO style)
            covar_matrix = target_posterior.covariance_matrix

            # Check if covariance is positive definite, fix if not
            if not is_pd(covar_matrix):
                covar_matrix = nearest_pd(covar_matrix)

            return GPyTorchPosterior(
                MultivariateNormal(final_mean.squeeze(-1), covar_matrix)
            )


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
        >>> # Create model and fit to source tasks
        >>> model = MHGPModel(input_dim=2)
        >>> source_data = [(X_source1, Y_source1), (X_source2, Y_source2)]
        >>> model.meta_fit(source_data)
        >>>
        >>> # Fit to target task
        >>> model.fit_target(X_target, Y_target)
        >>>
        >>> # Make predictions
        >>> posterior = model.posterior(X_test)
        >>> mean = posterior.mean
        >>> variance = posterior.variance
    """

    def __init__(self, input_dim: int) -> None:
        """Initialize the MHGP model.

        Args:
            input_dim: Dimensionality of the input space.
        """
        super().__init__()
        self.input_dim = input_dim

        self.source_gps: List[SingleTaskGP] = []
        """List of fitted source Gaussian Process models."""

        self.target_gp: Optional[SingleTaskGP] = None
        """The target Gaussian Process model."""

        self._fitted: bool = False
        """Whether the model has been fully fitted (including target task)."""

        self._source_data: List[Tuple[Tensor, Tensor]] = []
        """Cached source data for residual computation."""

    @property
    def num_outputs(self) -> int:
        """Number of outputs of the model."""
        return 1

    def meta_fit(self, source_data: List[Tuple[Tensor, Tensor]]) -> None:
        """Fit source GPs sequentially on residuals.

        This method implements the core MHGP training procedure by fitting each
        source GP to the residuals left by the previous GPs in the stack.

        Args:
            source_data: List of tuples containing (X, Y) data for each source task.
                Each X should have shape ``(n_points, input_dim)`` and each Y
                should have shape ``(n_points, 1)``.

        Raises:
            RuntimeError: If fitting any of the source GPs fails.
        """
        self._source_data = source_data

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

    def fit_target(self, X_target: Tensor, Y_target: Tensor) -> None:
        """Fit target GP on residuals from all source GPs.

        This method completes the MHGP training by fitting the final target GP
        to the residuals left after removing predictions from all source GPs.

        Args:
            X_target: Target task input data of shape ``(n_points, input_dim)``.
            Y_target: Target task output data of shape ``(n_points, 1)``.

        Raises:
            RuntimeError: If fitting the target GP fails.
        """
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
        """Compute posterior distribution by summing predictions from all GPs.

        The final prediction is the sum of predictions from all source GPs plus
        the target GP. The uncertainty estimate comes from the target GP only,
        following the original MHGP formulation.

        Args:
            X: Input locations at which to evaluate the model, with shape
                ``(batch_shape, n_points, input_dim)``.
            **kwargs: Additional keyword arguments (currently unused).

        Returns:
            A :class:`~botorch.posteriors.GPyTorchPosterior` object representing
            the posterior distribution at the input locations.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if not self._fitted:
            raise RuntimeError(
                "Model must be fitted first. Call meta_fit() and fit_target()."
            )

        with torch.no_grad():
            # Sum predictions from all source GPs
            total_mean = torch.zeros(X.shape[0], 1, dtype=X.dtype, device=X.device)

            for gp in self.source_gps:
                gp_posterior = gp.posterior(X)
                total_mean += gp_posterior.mean

            # Add target GP prediction
            target_posterior = self.target_gp.posterior(X)
            final_mean = total_mean + target_posterior.mean

            # Use target GP's covariance for uncertainty
            return GPyTorchPosterior(
                MultivariateNormal(
                    final_mean.squeeze(-1), target_posterior.covariance_matrix
                )
            )


def generate_task_data(
    n_points: int = 20, input_dim: int = 2, noise_std: float = 0.1
) -> Tuple[Tensor, Tensor, Tensor]:
    """Generate synthetic multi-task data following BoTorch conventions."""
    # Generate input points
    X = torch.rand(n_points * 3, input_dim, dtype=torch.float64)  # 3 tasks

    # Task indices (0, 1, 2)
    task_indices = torch.cat(
        [
            torch.zeros(n_points, 1, dtype=torch.float64),  # Task 0
            torch.ones(n_points, 1, dtype=torch.float64),  # Task 1
            torch.full((n_points, 1), 2.0, dtype=torch.float64),  # Task 2
        ]
    )

    # Generate task-specific functions
    def task_function(x: Tensor, task_id: int) -> Tensor:
        if task_id == 0:
            return torch.sin(3 * x[:, 0]) + 0.5 * x[:, 1]
        elif task_id == 1:
            return torch.cos(2 * x[:, 0]) + x[:, 1] ** 2
        else:  # task_id == 2 (target task)
            return torch.sin(2 * x[:, 0]) + torch.cos(x[:, 1]) + 0.3 * x[:, 0] * x[:, 1]

    # Generate Y values
    Y = torch.zeros(n_points * 3, 1, dtype=torch.float64)
    for i in range(3):
        start_idx = i * n_points
        end_idx = (i + 1) * n_points
        Y[start_idx:end_idx] = task_function(X[start_idx:end_idx], i).unsqueeze(-1)

    # Add noise
    Y += torch.randn_like(Y) * noise_std

    return X, Y, task_indices


if __name__ == "__main__":
    # Set double precision and seed
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(42)

    # Generate data with more points for stability
    X, Y, task_indices = generate_task_data(n_points=30, input_dim=2)

    # Split data by tasks
    n_points_per_task = 30
    X_source1 = X[:n_points_per_task]
    Y_source1 = Y[:n_points_per_task]

    X_source2 = X[n_points_per_task : 2 * n_points_per_task]
    Y_source2 = Y[n_points_per_task : 2 * n_points_per_task]

    X_target = X[2 * n_points_per_task :]
    Y_target = Y[2 * n_points_per_task :]

    print("Data shapes:")
    print(f"Source 1: X={X_source1.shape}, Y={Y_source1.shape}")
    print(f"Source 2: X={X_source2.shape}, Y={Y_source2.shape}")
    print(f"Target: X={X_target.shape}, Y={Y_target.shape}")

    # Prepare source data
    source_data = [(X_source1, Y_source1), (X_source2, Y_source2)]

    # Dictionary to store models and their training status
    models = {}

    # 1. Train Basic MHGP Model
    print("\n=== Training Basic MHGP Model ===")
    try:
        mhgp_model = MHGPModel(input_dim=2)
        mhgp_model.meta_fit(source_data)
        mhgp_model.fit_target(X_target, Y_target)
        models["MHGP Basic"] = {"model": mhgp_model, "trained": True, "error": None}
        print("Basic MHGP model trained successfully!")
    except Exception as e:
        models["MHGP Basic"] = {"model": None, "trained": False, "error": str(e)}
        print(f"Failed to train Basic MHGP model: {e}")

    # 2. Train Stable MHGP Model
    print("\n=== Training Stable MHGP Model ===")
    try:
        mhgp_stable_model = MHGPModelStable(input_dim=2)
        mhgp_stable_model.meta_fit(source_data)
        mhgp_stable_model.fit_target(X_target, Y_target)
        models["MHGP Stable"] = {
            "model": mhgp_stable_model,
            "trained": True,
            "error": None,
        }
        print("Stable MHGP model trained successfully!")
    except Exception as e:
        models["MHGP Stable"] = {"model": None, "trained": False, "error": str(e)}
        print(f"Failed to train Stable MHGP model: {e}")

    # 3. Train MultiTaskGP for comparison
    print("\n=== Training MultiTaskGP ===")
    try:
        X_multi = torch.cat([X, task_indices], dim=-1)
        multitask_model = MultiTaskGP(X_multi, Y, task_feature=-1)
        mll_multi = ExactMarginalLogLikelihood(
            multitask_model.likelihood, multitask_model
        )
        fit_gpytorch_mll(mll_multi)
        models["MultiTaskGP"] = {
            "model": multitask_model,
            "trained": True,
            "error": None,
        }
        print("MultiTaskGP model trained successfully!")
    except Exception as e:
        models["MultiTaskGP"] = {"model": None, "trained": False, "error": str(e)}
        print(f"Failed to train MultiTaskGP model: {e}")

    # 4. Compare predictions on test points
    print("\n=== Comparing Predictions ===")

    # Generate test points for target task
    X_test = torch.rand(10, 2, dtype=torch.float64)
    X_test_multi = torch.cat(
        [X_test, torch.full((10, 1), 2.0, dtype=torch.float64)], dim=-1
    )

    # Store predictions for each model
    predictions = {}

    for model_name, model_info in models.items():
        if not model_info["trained"]:
            print(f"Skipping {model_name} - training failed: {model_info['error']}")
            continue

        try:
            with torch.no_grad():
                if model_name == "MultiTaskGP":
                    posterior = model_info["model"].posterior(X_test_multi)
                else:
                    posterior = model_info["model"].posterior(X_test)

                mean = posterior.mean
                std = posterior.variance.sqrt()

                predictions[model_name] = {
                    "mean": mean,
                    "std": std,
                    "posterior": posterior,
                    "error": None,
                }

                print(f"{model_name} predictions - Mean: {mean[:5].flatten()}")
                print(f"{model_name} predictions - Std:  {std[:5].flatten()}")

        except Exception as e:
            predictions[model_name] = {
                "mean": None,
                "std": None,
                "posterior": None,
                "error": str(e),
            }
            print(f"Failed to get predictions from {model_name}: {e}")

    # 5. Visualization with error handling
    print("\n=== Visualization ===")

    # Create test grid for visualization
    x_test_1d = torch.linspace(0, 1, 50, dtype=torch.float64).unsqueeze(-1)
    x_test_2d = torch.cat([x_test_1d, torch.full_like(x_test_1d, 0.5)], dim=-1)
    x_test_multi_1d = torch.cat(
        [x_test_2d, torch.full((50, 1), 2.0, dtype=torch.float64)], dim=-1
    )

    # Get predictions for visualization
    viz_predictions = {}

    for model_name, model_info in models.items():
        if not model_info["trained"]:
            continue

        try:
            with torch.no_grad():
                if model_name == "MultiTaskGP":
                    pred = model_info["model"].posterior(x_test_multi_1d)
                else:
                    pred = model_info["model"].posterior(x_test_2d)

                viz_predictions[model_name] = pred

        except Exception as e:
            print(f"Failed to get visualization predictions from {model_name}: {e}")
            # Handle Cholesky errors specifically
            if "cholesky" in str(e).lower() or "positive-definite" in str(e).lower():
                print(f"Cholesky decomposition failed for {model_name}")
            viz_predictions[model_name] = None

    # Create plots for successfully trained models
    n_models = len([name for name, pred in viz_predictions.items() if pred is not None])

    if n_models > 0:
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
        if n_models == 1:
            axes = [axes]

        plot_idx = 0
        colors = ["blue", "green", "red", "orange", "purple"]

        for model_name, pred in viz_predictions.items():
            if pred is None:
                continue

            ax = axes[plot_idx]
            color = colors[plot_idx % len(colors)]

            try:
                # Plot mean and confidence interval
                mean_vals = pred.mean.numpy()
                std_vals = pred.variance.sqrt().numpy()

                ax.plot(
                    x_test_1d.numpy(),
                    mean_vals,
                    color=color,
                    label=f"{model_name} Mean",
                    linewidth=2,
                )
                ax.fill_between(
                    x_test_1d.flatten().numpy(),
                    (mean_vals - 2 * std_vals).flatten(),
                    (mean_vals + 2 * std_vals).flatten(),
                    alpha=0.3,
                    color=color,
                )

                # Plot target data
                ax.scatter(
                    X_target[:, 0].numpy(),
                    Y_target.flatten().numpy(),
                    c="red",
                    s=30,
                    label="Target Data",
                    alpha=0.7,
                )

                ax.set_title(f"{model_name}")
                ax.set_xlabel("X1")
                ax.set_ylabel("Y")
                ax.legend()
                ax.grid(True, alpha=0.3)

                plot_idx += 1

            except Exception as e:
                print(f"Failed to plot {model_name}: {e}")
                continue

        plt.tight_layout()
        plt.show()
    else:
        print(
            "No models available for visualization due to training/prediction failures."
        )

    # 6. Summary
    print("\n=== Training Summary ===")
    for model_name, model_info in models.items():
        status = (
            "✓ Success" if model_info["trained"] else f"✗ Failed: {model_info['error']}"
        )
        print(f"{model_name:15}: {status}")

    print("\n=== Prediction Summary ===")
    for model_name, pred_info in predictions.items():
        if pred_info.get("error"):
            status = f"✗ Failed: {pred_info['error']}"
        else:
            status = "✓ Success"
        print(f"{model_name:15}: {status}")

    print("\n=== Test completed! ===")
