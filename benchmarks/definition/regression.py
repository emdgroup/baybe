"""Regression benchmark configuration."""

import numpy as np
import torch
from attrs import define, field
from attrs.validators import instance_of
from scipy import stats
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from benchmarks.definition.base import Benchmark, BenchmarkSettings


# Custom function for log predictive density
def log_predictive_density(
    y_true: torch.Tensor, y_pred_mean: torch.Tensor, y_pred_var: torch.Tensor
) -> float:
    """Calculate the log predictive density for Gaussian predictions.

    Args:
        y_true: True target values
        y_pred_mean: Predicted mean values
        y_pred_var: Predicted variance values

    Returns:
        Average log predictive density
    """
    # For numerical stability, ensure variances are positive
    y_pred_var = np.maximum(y_pred_var, 1e-8)

    # Calculate log PDF for each prediction
    log_pdf = stats.norm.logpdf(y_true, loc=y_pred_mean, scale=np.sqrt(y_pred_var))

    # Return average log density
    return np.mean(log_pdf)


def negative_log_predictive_density(
    y_true: torch.Tensor, y_pred_mean: torch.Tensor, y_pred_var: torch.Tensor
) -> float:
    """Calculate the negative log predictive density (lower is better)."""
    return -log_predictive_density(y_true, y_pred_mean, y_pred_var)


# Dictionary mapping metric names to functions and whether they need variance
REGRESSION_METRICS = {
    "RMSE": {
        "function": mean_squared_error,
        "needs_variance": False,
        "transform": np.sqrt,
    },
    "R2": {"function": r2_score, "needs_variance": False, "transform": None},
    "MAE": {
        "function": mean_absolute_error,
        "needs_variance": False,
        "transform": None,
    },
    "MAX_ERROR": {"function": max_error, "needs_variance": False, "transform": None},
    "EXPLAINED_VARIANCE": {
        "function": explained_variance_score,
        "needs_variance": False,
        "transform": None,
    },
    "LPD": {
        "function": log_predictive_density,
        "needs_variance": True,
        "transform": None,
    },
    "NLPD": {
        "function": negative_log_predictive_density,
        "needs_variance": True,
        "transform": None,
    },
}

# Define whether higher values are better for each metric
METRICS_HIGHER_IS_BETTER = {
    "RMSE": False,
    "R2": True,
    "MAE": False,
    "MAX_ERROR": False,
    "EXPLAINED_VARIANCE": True,
    "LPD": True,
    "NLPD": False,
}


@define(frozen=True, kw_only=True)
class TransferLearningRegressionSettings(BenchmarkSettings):
    """Settings for transfer learning regression benchmark."""

    num_mc_iterations: int = field(validator=instance_of(int))
    """Number of Monte Carlo iterations."""

    input_dim: int = field(validator=instance_of(int), default=2)
    """Input dimension of the test functions."""

    max_train_points: int = field(validator=instance_of(int), default=10)
    """Maximum number of target training points to consider."""

    source_fractions: list[float] = field(default=[0.1, 0.3, 0.5, 0.7, 0.9])
    """Fractions of source data to use."""

    noise_std: float = field(validator=instance_of(float), default=0.1)
    """Standard deviation of noise to add to the data."""

    metrics: list[str] = field(default=["RMSE", "R2", "MAE"])
    """Metrics to evaluate. Must be keys in the REGRESSION_METRICS registry."""

    def __attrs_post_init__(self):
        # Validate that all specified metrics are available
        for metric in self.metrics:
            if metric not in REGRESSION_METRICS:
                raise ValueError(
                    f"Metric '{metric}' is not available."
                    "Available metrics: {list(REGRESSION_METRICS.keys())}"
                )


@define(frozen=True)
class TransferLearningRegression(Benchmark[TransferLearningRegressionSettings]):
    """Benchmark for comparing regression performance of vanilla GP vs TL models.

    Evaluates the predictive performance of transfer learning models compared
    to GP models. It generates synthetic data for source and target tasks, trains
    models with varying amounts of source and target data, evaluates their performance
    on held-out target data.
    """

    # No additional settings.
