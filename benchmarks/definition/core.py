"""Definition of the Benchmark class."""

import time
from collections.abc import Callable
from datetime import datetime
from typing import Generic, TypeVar
from uuid import UUID

from attrs import define, field
from attrs.validators import instance_of, is_callable
from pandas import DataFrame

from benchmarks.result.metadata_class import ResultMetadata
from benchmarks.result.result import Result

BenchmarkConfig = TypeVar("BenchmarkConfig")
BenchmarkFunction = Callable[[BenchmarkConfig], DataFrame]


@define
class Benchmark(Generic[BenchmarkConfig]):
    """A class to define a benchmark task."""

    name: str = field(validator=instance_of(str))
    """The name of the benchmark."""

    settings: BenchmarkConfig = field()
    """The configuration for the benchmark settings."""

    benchmark_callable: BenchmarkFunction = field(validator=is_callable())
    """The function that executes the benchmark code and returns
    the results as well as metadata."""

    identifier: UUID = field(validator=instance_of(UUID))
    """The unique identifier of the benchmark running which can be set
    to compare different executions of the same benchmark setting."""

    @property
    def description(self) -> str:
        """The description of the benchmark callable."""
        if self.benchmark_callable.__doc__ is None:
            return ""
        return self.benchmark_callable.__doc__

    def run(self) -> Result:
        """Execute the benchmark.

        The function will execute the benchmark
        and return the result
        """
        start_datetime = datetime.now()
        start_ns = time.time()
        result = self.benchmark_callable(self.settings)
        stop_ns = time.time()

        time_delta_sec = stop_ns - start_ns

        metadata = ResultMetadata(
            benchmark_name=self.name,
            execution_time_sec=time_delta_sec,
            start_datetime=start_datetime,
        )

        benchmark_result = Result(self.identifier, self.settings, result, metadata)
        return benchmark_result
