"""Regression benchmark configuration."""

from attrs import define, field
from attrs.validators import instance_of
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

from benchmarks.definition.base import Benchmark, BenchmarkSettings

# Dictionary mapping metric names to functions
REGRESSION_METRICS = {
    "RMSE": {"function": root_mean_squared_error},
    "MSE": {"function": mean_squared_error},
    "R2": {"function": r2_score},
    "MAE": {"function": mean_absolute_error},
    "MAX_ERROR": {"function": max_error},
    "EXPLAINED_VARIANCE": {"function": explained_variance_score},
}

# Define whether higher values are better for each metric
METRICS_HIGHER_IS_BETTER = {
    "RMSE": False,
    "MSE": False,
    "R2": True,
    "MAE": False,
    "MAX_ERROR": False,
    "EXPLAINED_VARIANCE": True,
}


def _validate_metrics(instance, attribute, value):
    """Validate that all specified metrics are available."""
    for metric in value:
        if metric not in REGRESSION_METRICS:
            raise ValueError(
                f"Metric '{metric}' is not available. "
                f"Available metrics: {list(REGRESSION_METRICS.keys())}"
            )
    return value


@define(frozen=True, kw_only=True)
class TransferLearningRegressionSettings(BenchmarkSettings):
    """Settings for transfer learning regression benchmark."""

    num_mc_iterations: int = field(validator=instance_of(int))
    """Number of Monte Carlo iterations."""

    max_train_points: int = field(validator=instance_of(int), default=10)
    """Maximum number of target training points to consider."""

    source_fractions: list[float] = field(default=[0.1, 0.3, 0.5, 0.7, 0.9])
    """Fractions of source data to use."""

    noise_std: float = field(validator=instance_of(float), default=0.1)
    """Standard deviation of noise to add to the data."""

    metrics: list[str] = field(
        default=["RMSE", "R2", "MAE"], validator=_validate_metrics
    )
    """Metrics to evaluate. Must be keys in the REGRESSION_METRICS registry."""


@define(frozen=True)
class TransferLearningRegression(Benchmark[TransferLearningRegressionSettings]):
    """Benchmark for comparing regression performance of vanilla GP vs TL models.

    Evaluates the predictive performance of transfer learning models compared
    to GP models. It generates synthetic data for source and target tasks, trains
    models with varying amounts of source and target data, evaluates their performance
    on held-out target data.
    """
