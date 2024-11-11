"""Benchmark definition functionality."""

import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Generic
from uuid import UUID

from attrs import define, field
from attrs.validators import instance_of, is_callable
from pandas import DataFrame

from benchmarks.definition.config import BenchmarkConfig
from benchmarks.result.metadata import ResultMetadata
from benchmarks.result.result import Result

BenchmarkFunction = Callable[[BenchmarkConfig], DataFrame]


@define
class Benchmark(Generic[BenchmarkConfig]):
    """Definition of a benchmark task."""

    identifier: UUID = field(validator=instance_of(UUID))
    """The unique identifier of the benchmark.
    Can be used to compare different runs of the same benchmark."""

    name: str = field(validator=instance_of(str))
    """The name of the benchmark."""

    callable: BenchmarkFunction = field(validator=is_callable())
    """The callable that executes the benchmark code and returns the result."""

    settings: BenchmarkConfig | None = field(default=None)
    """The benchmark configuration."""

    @property
    def description(self) -> str:
        """The description of the benchmark callable."""
        if self.callable.__doc__ is None:
            return ""
        return self.callable.__doc__

    def __call__(self) -> Result:
        """Execute the benchmark and return the result."""
        start_datetime = datetime.now(timezone.utc)

        start_sec = time.perf_counter()
        result = self.callable(self.settings)
        stop_sec = time.perf_counter()

        metadata = ResultMetadata(
            start_datetime=start_datetime,
            execution_time_sec=stop_sec - start_sec,
        )

        return Result(self.identifier, result, metadata)
