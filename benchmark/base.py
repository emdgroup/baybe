"""Base class for benchmarking implementations."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from uuid import UUID, uuid4

from attrs import define, field
from pandas import DataFrame

from benchmark.metric import Metric
from benchmark.result import Result


@define
class Benchmark(ABC):
    """Abstract base class for all benchmarks."""

    title: str
    """The title of the benchmark."""

    benchmark_function: Callable[[], DataFrame]
    """The function that executes the benchmark code and returns the results."""

    identifier: UUID = field(factory=uuid4)
    """The unique identifier of the benchmark running which can be set
    to compare different executions of the same benchmark setting."""

    metadata: dict[str, str] = field(factory=lambda: dict())
    """Metadata about the benchmark."""

    metrics: list[Metric] = field(factory=lambda: list())
    """Optional metrics to evaluate the benchmarking results."""

    def __str__(self) -> str:
        return self.title

    def __repr__(self) -> str:
        classname = self.__class__.__name__
        return f"{classname}(name={self.title}, identifier={self.identifier})"

    @abstractmethod
    def execute_benchmark(self) -> Result:
        """Execute the benchmark and return the results."""
        pass

    @abstractmethod
    def get_results(self) -> Result:
        """Return the results of the benchmark."""
        pass
