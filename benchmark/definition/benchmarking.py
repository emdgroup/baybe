"""Definition of the Benchmark class."""

import time
from collections.abc import Callable
from uuid import UUID, uuid4

from attrs import define, field
from attrs.validators import instance_of
from pandas import DataFrame

from benchmark.result.result import Result


@define
class Benchmark:
    """A class to define a benchmark task."""

    name: str = field(validator=instance_of(str))
    """The name of the benchmark."""

    benchmark_function: Callable[[], tuple[DataFrame, dict[str, str]]]
    """The function that executes the benchmark code and returns
    the results as well as metadata."""

    identifier: UUID = field(factory=uuid4, validator=instance_of(UUID))
    """The unique identifier of the benchmark running which can be set
    to compare different executions of the same benchmark setting."""

    def run(self) -> Result:
        """Execute the benchmark.

        The function will execute the benchmark and return the result apply added
        metrics if any were set and measures execution time.
        """
        start_ns = time.perf_counter_ns()
        result, metadata = self.benchmark_function()
        stop_ns = time.perf_counter_ns()
        time_delta = stop_ns - start_ns
        benchmark_result = Result(
            self.name, self.identifier, metadata, result, time_delta
        )
        return benchmark_result
