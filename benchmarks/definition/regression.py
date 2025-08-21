"""Regression benchmark configuration."""

from attrs import define, field
from attrs.validators import and_, deep_iterable, ge, instance_of, le

from baybe.utils.validation import finite_float
from benchmarks.definition.base import Benchmark, BenchmarkSettings


@define(frozen=True, kw_only=True)
class RegressionBenchmarkSettings(BenchmarkSettings):
    """Base settings for regression benchmarks."""

    n_mc_iterations: int = field(validator=instance_of(int))
    """Number of Monte Carlo iterations."""

    max_n_train_points: int = field(validator=instance_of(int))
    """Maximum number of training points to consider."""

    noise_std: float = field(converter=float, validator=and_(finite_float, ge(0.0)))
    """Standard deviation of noise to add to the data."""


@define(frozen=True, kw_only=True)
class TransferLearningRegressionBenchmarkSettings(RegressionBenchmarkSettings):
    """Settings for transfer learning regression benchmark."""

    source_fractions: tuple[float, ...] = field(
        validator=deep_iterable(
            member_validator=and_(ge(0.0), le(1.0)),
            iterable_validator=instance_of(tuple),
        )
    )
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
    Benchmark[TransferLearningRegressionBenchmarkSettings]
):
    """Benchmark for comparing regression performance of non-TL vs TL models.

    Evaluates the predictive performance of transfer learning models compared
    to GP models. It generates data for source and target tasks, trains models
    with varying amounts of source and target data, evaluates their performance
    on held-out target data.
    """
