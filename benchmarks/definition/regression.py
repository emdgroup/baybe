"""Regression benchmark configuration."""

from attrs import define, field
from attrs.validators import instance_of

from benchmarks.definition.base import Benchmark, BenchmarkSettings


@define(frozen=True, kw_only=True)
class TransferLearningRegressionSettings(BenchmarkSettings):
    """Settings for transfer learning regression benchmark."""

    n_mc_iterations: int = field(validator=instance_of(int))
    """Number of Monte Carlo iterations."""

    max_n_train_points: int = field(validator=instance_of(int), default=10)
    """Maximum number of target training points to consider."""

    source_fractions: list[float] = field(default=[0.1, 0.3, 0.5, 0.7, 0.9])
    """Fractions of source data to use."""

    noise_std: float = field(default=0.1, converter=float)
    """Standard deviation of noise to add to the data."""


@define(frozen=True)
class TransferLearningRegression(Benchmark[TransferLearningRegressionSettings]):
    """Benchmark for comparing regression performance of non-TL vs TL models.

    Evaluates the predictive performance of transfer learning models compared
    to GP models. It generates synthetic data for source and target tasks, trains
    models with varying amounts of source and target data, evaluates their performance
    on held-out target data.
    """
