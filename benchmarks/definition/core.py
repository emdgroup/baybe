"""Definition of the Benchmark class."""

import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Generic
from uuid import UUID

from attrs import define, field
from attrs.validators import instance_of, is_callable
from pandas import DataFrame

from benchmarks.definition.config import BenchmarkConfig
from benchmarks.result.metadata_class import ResultMetadata
from benchmarks.result.result import Result

BenchmarkFunction = Callable[[BenchmarkConfig], DataFrame]


@define
class Benchmark(Generic[BenchmarkConfig]):
    """A class to define a benchmark task."""

    identifier: UUID = field(validator=instance_of(UUID))
    """The unique identifier of the benchmark running which can be set
    to compare different executions of the same benchmark setting."""

    name: str = field(validator=instance_of(str))
    """The name of the benchmark."""

    callable: BenchmarkFunction = field(validator=is_callable())
    """The function that executes the benchmark code and returns
    the results as well as metadata."""

    settings: BenchmarkConfig | None = field(default=None)
    """The configuration for the benchmark settings."""

    @property
    def description(self) -> str:
        """The description of the benchmark callable."""
        if self.callable.__doc__ is None:
            return ""
        return self.callable.__doc__

    def __call__(self) -> Result:
        """Execute the benchmark.

        The function will execute the benchmark
        and return the result
        """
        start_datetime = datetime.now(timezone.utc)
        start_sec = time.perf_counter()
        result = self.callable(self.settings)
        stop_sec = time.perf_counter()

        time_delta_sec = start_sec - stop_sec

        metadata = ResultMetadata(
            execution_time_sec=time_delta_sec,
            start_datetime=start_datetime,
        )

        benchmark_result = Result(self.identifier, result, metadata)
        return benchmark_result
