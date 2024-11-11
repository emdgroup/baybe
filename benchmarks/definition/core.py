"""Benchmark definition functionality."""

import time
from datetime import datetime, timedelta, timezone

from attrs import define, field
from attrs.validators import instance_of

from benchmarks.definition.config import BenchmarkExecutableBase
from benchmarks.result.metadata import ResultMetadata
from benchmarks.result.result import Result


@define
class BenchmarkDefinition:
    """Definition of a benchmark task."""

    identifier: str = field(validator=instance_of(str))
    """The unique identifier of the benchmark.
    Can be used to compare different runs of the same benchmark."""

    benchmark_function_definition: BenchmarkExecutableBase = field(
        validator=instance_of(BenchmarkExecutableBase)
    )
    """The benchmark function definition which contains information about the callable
    holding the domain and further information about it."""

    @identifier.validator
    def _validate_identifier(self, _, value: str) -> None:
        """Validate the identifier."""
        LOWER_CASE_ALPHABETIC_CHARS = "abcdefghijklmnopqrstuvwxyz"
        SAFE_CHARS = (
            "0123456789"
            + LOWER_CASE_ALPHABETIC_CHARS
            + LOWER_CASE_ALPHABETIC_CHARS.upper()
            + "!-_.()"
        )

        if not all(char in SAFE_CHARS for char in value):
            raise ValueError(
                f"Invalid identifier '{value}'. "
                f"Only the following characters are allowed: {SAFE_CHARS}"
            )

    def __call__(self) -> Result:
        """Execute the benchmark and return the result."""
        start_datetime = datetime.now(timezone.utc)

        start_sec = time.perf_counter()
        result = self.benchmark_function_definition()
        stop_sec = time.perf_counter()

        duration = timedelta(seconds=stop_sec - start_sec)

        metadata = ResultMetadata(
            start_datetime=start_datetime,
            duration=duration,
        )

        return Result(self.identifier, result, metadata)
