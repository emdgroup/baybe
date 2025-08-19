"""Regression benchmark configuration."""

from attrs import define, field
from attrs.validators import instance_of

from benchmarks.definition.base import Benchmark, BenchmarkSettings


@define(frozen=True, kw_only=True)
class RegressionBenchmarkSettings(BenchmarkSettings):
    """Base settings for regression benchmarks."""

    n_mc_iterations: int = field(validator=instance_of(int))
    """Number of Monte Carlo iterations."""

    max_n_train_points: int = field(validator=instance_of(int), default=10)
    """Maximum number of training points to consider."""

    noise_std: float = field(default=0.1, converter=float)
    """Standard deviation of noise to add to the data."""


@define(frozen=True, kw_only=True)
class TransferLearningRegressionBenchmarkSettings(RegressionBenchmarkSettings):
    """Settings for transfer learning regression benchmark."""

    source_fractions: tuple[float, ...]
    """Fractions of source data to use."""


@define(frozen=True)
class RegressionBenchmark(Benchmark[RegressionBenchmarkSettings]):
    """Base class for regression benchmarks.

    Provides common infrastructure for benchmarking models on regression tasks,
    including Monte Carlo evaluation, train/test splitting, and performance
    metric calculation.
    """


@define(frozen=True)
class TransferLearningRegressionBenchmark(
    RegressionBenchmark[TransferLearningRegressionBenchmarkSettings]
):
    """Benchmark for comparing regression performance of non-TL vs TL models.

    Evaluates the predictive performance of transfer learning models compared
    to GP models. It generates synthetic data for source and target tasks, trains
    models with varying amounts of source and target data, evaluates their performance
    on held-out target data.
    """
