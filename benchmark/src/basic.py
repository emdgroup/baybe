"""A basic benchmarking class for testing."""

import concurrent.futures
import os
import time

from attrs import define, field
from pandas import DataFrame

from benchmark.src.base import Benchmark
from benchmark.src.result.basic import MultiResult, SingleResult


@define
class SingleExecutionBenchmark(Benchmark):
    """A basic benchmarking class for testing a single benchmark execution."""

    _benchmark_result: SingleResult = field(default=None)
    """The result of the benchmarking which is set after execution."""

    def execute_benchmark(self) -> SingleResult:
        """Execute the benchmark.

        The function will execute the benchmark and return
        the result apply added metrics if any were set and
        measures execution time.
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
        for metric in self.metrics:
            self._benchmark_result.evaluate_result(metric, self.objective_scenarios)
        return self._benchmark_result

    def get_result(self) -> SingleResult:
        """Return the single result of the benchmark."""
        if not self._benchmark_result:
            self._benchmark_result = self.execute_benchmark()
        return self._benchmark_result


@define
class MultiExecutionBenchmark(Benchmark):
    """Benchmarking class for testing multiple executions of the same scenario."""

    number_of_runs: int = field(default=3)
    """The number of times to run the benchmark. Default is 3."""

    _benchmark_results: MultiResult = field(default=None)
    """The results of the benchmarking which is set after execution."""

    def _execute_with_timing(self) -> tuple[DataFrame, int, dict[str, str]]:
        """Execute the benchmark.

        The function will execute the benchmark and return
        the result apply added metrics if any were set and
        measures execution time.
        """
        try:
            start_ns = time.perf_counter_ns()
            result, metadata = self.benchmark_function()
            stop_ns = time.perf_counter_ns()
        except Exception as e:
            raise Exception(f"Error in benchmark {self.identifier}: {e}")
        time_delta = stop_ns - start_ns
        return result, time_delta, metadata

    def execute_benchmark(self) -> MultiResult:
        """Execute the benchmark in parallel.

        This function will execute the benchmark in parallel using
        the number of cores available on the machine. The function
        will return the results of the benchmark and execute the
        metrics if any were set. Those are calculating the arithmetic mean
        of all single results by default. See
        :class:benchmark.src.result.basic.MultiResult for more information.
        """
        num_cores = os.cpu_count()
        results: list[SingleResult] = []
        number_of_iterations = range(self.number_of_runs)
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = [
                executor.submit(self._execute_with_timing) for _ in number_of_iterations
            ]
            for future in concurrent.futures.as_completed(futures):
                result_benchmarking, time_delta, metadata = future.result()
                results.append(
                    SingleResult(
                        self.title,
                        self.identifier,
                        metadata,
                        result_benchmarking,
                        time_delta,
                    )
                )
        self._metadata = results[0].metadata
        self._benchmark_results = MultiResult(
            self.title, self.identifier, self._metadata, results
        )
        for metric in self.metrics:
            self._benchmark_results.evaluate_result(metric, self.objective_scenarios)
        return self._benchmark_results

    def get_result(self) -> MultiResult:
        """Return the benchmark's result or creates the result is not present jet."""
        if not self._benchmark_results:
            self._benchmark_results = self.execute_benchmark()
        return self._benchmark_results
