"""Definition of the Benchmark class."""

import time
from collections.abc import Callable
from datetime import datetime
from typing import Any
from uuid import UUID

from attrs import define, field
from attrs.validators import instance_of, is_callable
from pandas import DataFrame

from benchmark.result.metadata_class import ResultMetadata
from benchmark.result.result import Result

BenchmarkFunction = Callable[[], tuple[DataFrame, dict[str, Any]]]


@define
class Benchmark:
    """A class to define a benchmark task."""

    name: str = field(validator=instance_of(str))
    """The name of the benchmark."""

    benchmark_function: BenchmarkFunction = field(validator=is_callable())
    """The function that executes the benchmark code and returns
    the results as well as metadata."""

    identifier: UUID = field(validator=instance_of(UUID))
    """The unique identifier of the benchmark running which can be set
    to compare different executions of the same benchmark setting."""

    @property
    def description(self) -> str:
        """The description of the benchmark callable."""
        if self.benchmark_function.__doc__ is None:
            return ""
        return self.benchmark_function.__doc__

    def run(self) -> Result:
        """Execute the benchmark.

        The function will execute the benchmark
        and return the result
        """
        start_datetime = datetime.now()
        start_ns = time.perf_counter_ns()
        result, benchmark_settings = self.benchmark_function()
        stop_ns = time.perf_counter_ns()

        benchmark_settings["benchmark_name"] = self.name
        time_delta = stop_ns - start_ns
        time_delta_sec = time_delta / 1e9

        result_metadata = ResultMetadata(
            benchmark_name=self.name,
            execution_time_sec=time_delta_sec,
            start_datetime=start_datetime,
        )

        benchmark_result = Result(
            self.identifier, benchmark_settings, result, result_metadata
        )
        return benchmark_result
