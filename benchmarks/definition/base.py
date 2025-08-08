"""Basic benchmark configuration."""

import logging
import sys
import time
from abc import ABC
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Generic, TypeVar

import attr
from attrs import define, field
from attrs.validators import instance_of
from cattrs import override
from cattrs.gen import make_dict_unstructure_fn
from pandas import DataFrame

from benchmarks import ConvergenceBenchmarkSettings
from benchmarks.result import Result, ResultMetadata
from benchmarks.serialization import BenchmarkSerialization, converter

logger = logging.getLogger(__name__)

stdout = logging.StreamHandler(stream=sys.stdout)

logger.addHandler(stdout)
logger.setLevel(logging.INFO)


@define(frozen=True, kw_only=True)
class BenchmarkSettings(ABC, BenchmarkSerialization):
    """The basic benchmark configuration."""

    random_seed: int = field(validator=instance_of(int), default=1337)
    """The used random seed."""


BenchmarkSettingsType = TypeVar("BenchmarkSettingsType", bound=BenchmarkSettings)


@define(frozen=True)
class Benchmark(Generic[BenchmarkSettingsType], BenchmarkSerialization):
    """The base class for all benchmark definitions."""

    function: Callable[[BenchmarkSettingsType], DataFrame] = field()
    """The callable containing the benchmarking logic."""

    settings: BenchmarkSettingsType = field()
    """The benchmark configuration."""

    @function.validator
    def _validate_function(self, _, function) -> None:
        if function.__doc__ is None:
            raise ValueError("The benchmark function must have a docstring.")

    @property
    def name(self) -> str:
        """The name of the benchmark function."""
        return self.function.__name__

    @property
    def description(self) -> str:
        """The description of the benchmark function."""
        assert self.function.__doc__ is not None
        return self.function.__doc__

    def __call__(self, smoketest: str | None = None) -> Result:
        """Execute the benchmark and return the result."""
        settings = self.settings
        if smoketest == "runthrough" and isinstance(
            self.settings, ConvergenceBenchmarkSettings
        ):
            settings = attr.evolve(self.settings, n_doe_iterations=2, n_mc_iterations=1)

        start_datetime = datetime.now(timezone.utc)

        logger.info(
            "=" * 80
            + f"\nRunning benchmark '{self.name}' with "
            + f"random seed {settings.random_seed}.\n"
        )

        start_sec = time.perf_counter()
        result = self.function(settings)
        stop_sec = time.perf_counter()

        duration = timedelta(seconds=stop_sec - start_sec)

        logger.info(
            f"\nFinished benchmark '{self.name}' after {duration} "
            + f"with random seed {settings.random_seed}.\n"
            + "=" * 80
        )

        metadata = ResultMetadata(
            start_datetime=start_datetime,
            duration=duration,
        )

        return Result(self.name, result, metadata)


@converter.register_unstructure_hook
def unstructure_benchmark(benchmark: Benchmark) -> dict:
    """Unstructure a benchmark instance."""
    fn = make_dict_unstructure_fn(
        type(benchmark), converter, function=override(omit=True)
    )
    return {
        "name": benchmark.name,
        "description": benchmark.description,
        **fn(benchmark),
    }
