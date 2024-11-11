"""Definition of the Benchmark class."""

import time
from collections.abc import Callable
from uuid import UUID, uuid4

from attrs import define, field
from pandas import DataFrame

from benchmark.result.result import Result


@define
class Benchmark:
    """A class to define a benchmark task."""

    name: str
    """The name of the benchmark."""

    benchmark_function: Callable[[], tuple[DataFrame, dict[str, str]]]
    """The function that executes the benchmark code and returns
    the results as well as metadata."""

    objective_scenarios: list[str] = field(factory=list)
    """The name of the simulated scenarios referring to
    :func:`baybe.simulation.core.simulate_experiment`: to
    evaluate the benchmarking results."""

    identifier: UUID = field(factory=uuid4)
    """The unique identifier of the benchmark running which can be set
    to compare different executions of the same benchmark setting."""

    _metadata: dict[str, str] = field(init=False)
    """Metadata about the benchmark. Will be set after the benchmark is executed."""

    _benchmark_result: Result = field(init=False)
    """The result of the benchmarking which is set after execution.
    Set after the benchmark is executed."""

    def run(self) -> Result:
        """Execute the benchmark.

        The function will execute the benchmark and return the result apply added
        metrics if any were set and measures execution time.
        """
        start_ns = time.perf_counter_ns()
        result, self._metadata = self.benchmark_function()
        stop_ns = time.perf_counter_ns()
        time_delta = stop_ns - start_ns
        self._benchmark_result = Result(
            self.name, self.identifier, self._metadata, result, time_delta
        )
        return self._benchmark_result
