"""Definition of the Benchmark class."""

import time
from collections.abc import Callable
from typing import Any
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

    benchmark_function: Callable[[], tuple[DataFrame, dict[str, Any]]]
    """The function that executes the benchmark code and returns
    the results as well as metadata."""

    identifier: UUID = field(factory=uuid4, validator=instance_of(UUID))
    """The unique identifier of the benchmark running which can be set
    to compare different executions of the same benchmark setting."""

    description: str = field(validator=instance_of(str), init=False)
    """The description of the benchmark callable. Will be
    set as the docstring of the benchmark function automatically."""

    def __attrs_post_init__(self):
        if self.benchmark_function.__doc__ is None:
            raise ValueError("Description of the benchmark callable is not set.")
        self.description = self.benchmark_function.__doc__

    def run(self) -> Result:
        """Execute the benchmark.

        The function will execute the benchmark
        and return the result
        """
        start_ns = time.perf_counter_ns()
        result, metadata = self.benchmark_function()
        stop_ns = time.perf_counter_ns()

        metadata["benchmark_name"] = self.name
        time_delta = stop_ns - start_ns
        time_delta_sec = time_delta / 1e9
        benchmark_result = Result(self.identifier, metadata, result, time_delta_sec)
        return benchmark_result
