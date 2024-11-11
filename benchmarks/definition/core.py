"""Benchmark definition functionality."""

import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Generic
from uuid import UUID

from attrs import define, field
from attrs.validators import instance_of, is_callable
from pandas import DataFrame

from benchmarks.definition.config import BenchmarkSettings
from benchmarks.result.metadata import ResultMetadata
from benchmarks.result.result import Result

BenchmarkFunction = Callable[[BenchmarkSettings], DataFrame]


@define
class Benchmark(Generic[BenchmarkSettings]):
    """Definition of a benchmark task."""

    identifier: UUID = field(validator=instance_of(UUID))
    """The unique identifier of the benchmark.
    Can be used to compare different runs of the same benchmark."""

    name: str = field(validator=instance_of(str))
    """The name of the benchmark."""

    callable: BenchmarkFunction = field(validator=is_callable())
    """The callable that executes the benchmark code and returns the result."""

    description: str = field(validator=instance_of(str))
    """The description of the benchmark callable."""

    best_possible_result: float | None = field(
        validator=instance_of((float, None)), default=None
    )
    """The best possible result which can be achieved in the optimization process."""

    settings: BenchmarkSettings | None = field(default=None)
    """The benchmark configuration."""

    def __call__(self) -> Result:
        """Execute the benchmark and return the result."""
        start_datetime = datetime.now(timezone.utc)

        start_sec = time.perf_counter()
        result = self.callable(self.settings)
        stop_sec = time.perf_counter()

        duration = timedelta(seconds=stop_sec - start_sec)

        metadata = ResultMetadata(
            start_datetime=start_datetime,
            duration=duration,
            best_possible_result=self.best_possible_result,
        )

        return Result(self.identifier, result, metadata)
