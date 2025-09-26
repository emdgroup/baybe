"""Regression benchmark configuration."""

from attrs import define, field, fields

from benchmarks.definition.base import (
    Benchmark,
    BenchmarkSettings,
    RunMode,
    make_runmode_attr_validator,
)


@define(frozen=True, kw_only=True)
class RegressionBenchmarkSettings(BenchmarkSettings):
    """Base settings for regression benchmarks."""

    n_mc_iterations_settings: dict[RunMode, int] = field(
        validator=make_runmode_attr_validator(int)
    )
    """The settings for number of Monte Carlo iterations."""

    max_n_train_points_settings: dict[RunMode, int] = field(
        validator=make_runmode_attr_validator(int)
    )
    """The settings for maximum number of training points to consider."""

    noise_std_settings: dict[RunMode, float] = field(
        validator=make_runmode_attr_validator(float)
    )
    """The settings for standard deviation of noise on the target training data."""

    @property
    def n_mc_iterations(self) -> int:
        """The number of Monte Carlo iterations for the current runmode."""
        if self.runmode not in self.n_mc_iterations_settings:
            raise ValueError(
                "Current runmode not found in "
                f"{fields(RegressionBenchmarkSettings).n_mc_iterations_settings.name}"
                f": {self.runmode.value}"
            )
        return self.n_mc_iterations_settings[self.runmode]

    @property
    def max_n_train_points(self) -> int:
        """The maximum number of training points for the current runmode."""
        if self.runmode not in self.max_n_train_points_settings:
            raise ValueError(
                "Current runmode not found in "
                f"{fields(RegressionBenchmarkSettings).max_n_train_points_settings.name}"
                f": {self.runmode.value}"
            )
        return self.max_n_train_points_settings[self.runmode]

    @property
    def noise_std(self) -> float:
        """The standard deviation of noise for the current runmode."""
        if self.runmode not in self.noise_std_settings:
            raise ValueError(
                "Current runmode not found in "
                f"{fields(RegressionBenchmarkSettings).noise_std_settings.name}"
                f": {self.runmode.value}"
            )
        return self.noise_std_settings[self.runmode]


@define(frozen=True, kw_only=True)
class TransferLearningRegressionBenchmarkSettings(RegressionBenchmarkSettings):
    """Settings for transfer learning regression benchmark."""

    source_fractions_settings: dict[RunMode, tuple[float, ...]] = field(
        validator=make_runmode_attr_validator(tuple)
    )
    """The settings for fractions of source data to use."""

    stratified_source_sampling_settings: dict[RunMode, bool] = field(
        validator=make_runmode_attr_validator(bool)
    )
    """The settings for whether to use stratified sampling across source tasks."""

    @property
    def source_fractions(self) -> tuple[float, ...]:
        """The fractions of source data to use."""
        if self.runmode not in self.source_fractions_settings:
            raise ValueError(
                "Current runmode not found in "
                f"{fields(TransferLearningRegressionBenchmarkSettings).source_fractions_settings.name}"
                f": {self.runmode.value}"
            )
        return self.source_fractions_settings[self.runmode]

    @property
    def stratified_source_sampling(self) -> bool:
        """Whether to use stratified sampling across source tasks."""
        if self.runmode not in self.stratified_source_sampling_settings:
            raise ValueError(
                "Current runmode not found in "
                f"{fields(TransferLearningRegressionBenchmarkSettings).stratified_source_sampling_settings.name}"
                f": {self.runmode.value}"
            )
        return self.stratified_source_sampling_settings[self.runmode]


@define(frozen=True)
class RegressionBenchmark(Benchmark[RegressionBenchmarkSettings]):
    """Base class for regression benchmarks.

    Provides common infrastructure for benchmarking models on regression tasks,
    including Monte Carlo evaluation, train/test splitting, and performance
    metric calculation.
    """

    settings: RegressionBenchmarkSettings


@define(frozen=True)
class TransferLearningRegressionBenchmark(RegressionBenchmark):
    """Benchmark for comparing regression performance of non-TL vs TL models.

    Evaluates the predictive performance of transfer learning models compared
    to GP models. It generates data for source and target tasks, trains models
    with varying amounts of source and target data, evaluates their performance
    on held-out target data.
    """

    settings: TransferLearningRegressionBenchmarkSettings
