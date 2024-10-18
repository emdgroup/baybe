"""A basic benchmarking class for testing."""

import concurrent.futures
import os
import time

from attrs import define, field
from pandas import DataFrame
from typing_extensions import override

from src.base import Benchmark
from src.result.basic import MultiResult, SingleResult


@define
class SingleExecutionBenchmark(Benchmark):
    """A basic benchmarking class for testing a single benchmark execution."""

    _benchmark_result: SingleResult = field(default=None)
    """The result of the benchmarking which is set after execution."""

    @override
    def execute_benchmark(self) -> SingleResult:
        """Execute the benchmark in parallel."""
        start_ns = time.perf_counter_ns()
        result, self.metadata = self.benchmark_function()
        stop_ns = time.perf_counter_ns()
        time_delta = stop_ns - start_ns
        self._benchmark_result = SingleResult(
            self.title, self.identifier, self.metadata, result, time_delta
        )
        for metric in self.metrics:
            self._benchmark_result.evaluate_result(metric)
        return self._benchmark_result

    @override
    def get_result(self) -> SingleResult:
        """Return the results of the benchmark."""
        if not self._benchmark_result:
            self._benchmark_result = self.execute()
        return self._benchmark_result


@define
class MultiExecutionBenchmark(Benchmark):
    """Benchmarking class for testing multiple benchmark executions."""

    number_of_runs: int = field(default=3)
    """The number of times to run the benchmark. Default is 3."""

    _benchmark_results: MultiResult = field(default=None)
    """The results of the benchmarking which is set after execution."""

    def _execute_with_timing(self) -> tuple[DataFrame, int]:
        """Execute the benchmark and return the execution time."""
        start_ns = time.perf_counter_ns()
        result, self.metadata = self.benchmark_function()
        stop_ns = time.perf_counter_ns()
        time_delta = stop_ns - start_ns
        return result, time_delta

    @override
    def execute_benchmark(self) -> MultiResult:
        """Execute the benchmark in parallel."""
        num_cores = os.cpu_count()
        results: list[SingleResult] = []
        number_of_iterations = range(self.number_of_runs)
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = [
                executor.submit(self._execute_with_timing) for _ in number_of_iterations
            ]
            for future in concurrent.futures.as_completed(futures):
                result_benchmarking, time_delta = future.result()
                results.append(
                    SingleResult(
                        self.title,
                        self.identifier,
                        self.metadata,
                        result_benchmarking,
                        time_delta,
                    )
                )
        self._benchmark_results = MultiResult(
            self.title, self.identifier, self.metadata, results
        )
        for metric in self.metrics:
            self._benchmark_results.evaluate_result(metric)
        return self._benchmark_results

    @override
    def get_result(self) -> MultiResult:
        """Return the results of the benchmark."""
        if not self._benchmark_results:
            self._benchmark_results = self.execute()
        return self._benchmark_results
