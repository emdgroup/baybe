"""A basic benchmarking class for testing."""

import time

from attrs import define, field
from typing_extensions import override

from benchmark.definition.base import Benchmark
from benchmark.result.basic_results import SingleResult


@define
class SingleExecutionBenchmark(Benchmark):
    """A basic benchmarking class for testing a single benchmark execution."""

    _benchmark_result: SingleResult = field(default=None)
    """The result of the benchmarking which is set after execution."""

    @override
    def execute_benchmark(self) -> SingleResult:
        """Execute the benchmark.

        The function will execute the benchmark and return the result apply added
        metrics if any were set and measures execution time.
        """
        try:
            start_ns = time.perf_counter_ns()
            result, self._metadata = self.benchmark_function()
            stop_ns = time.perf_counter_ns()
        except Exception as e:
            raise Exception(f"Error in benchmark {self.identifier}: {e}")
        time_delta = stop_ns - start_ns
        self._benchmark_result = SingleResult(
            self.title, self.identifier, self._metadata, result, time_delta
        )
        return self._benchmark_result
