"""Base classes for benchmarking metrics."""

from abc import ABC, abstractmethod

from benchmark.result import Result


class Metric(ABC):
    """Abstract base class for all benchmarking metrics."""

    @abstractmethod
    def evaluate(self, result: Result) -> float:
        """Evaluate the benchmarking metric and return the result."""
        pass
