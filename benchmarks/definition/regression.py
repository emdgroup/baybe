"""Regression benchmark configuration."""

import numpy as np
from attrs import define, field
from attrs.validators import instance_of
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from benchmarks.definition.base import Benchmark, BenchmarkSettings

# Dictionary mapping metric names to functions
REGRESSION_METRICS = {
    "RMSE": {
        "function": mean_squared_error,
        "transform": np.sqrt,
    },
    "R2": {"function": r2_score, "transform": None},
    "MAE": {
        "function": mean_absolute_error,
        "transform": None,
    },
    "MAX_ERROR": {"function": max_error, "transform": None},
    "EXPLAINED_VARIANCE": {
        "function": explained_variance_score,
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
                    f"Metric '{metric}' is not available. "
                    f"Available metrics: {list(REGRESSION_METRICS.keys())}"
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
