"""Basic result classes for benchmarking."""

from typing import Any
from uuid import UUID

from attrs import define, field
from attrs.validators import instance_of
from pandas import DataFrame


@define(frozen=True)
class Result:
    """A single result of the benchmarking."""

    name: str = field(validator=instance_of(str))
    """The name of the benchmarking result."""

    identifier: UUID = field(validator=instance_of(UUID))
    """The unique identifier of the benchmark running which can be set
    to compare different executions of the same benchmark setting."""

    metadata: dict[str, str] = field()
    """Metadata about the benchmarking result."""

    benchmark_result: DataFrame = field()
    """The result of the benchmarked callable."""

    execution_time_ns: int = field(validator=instance_of(int))
    """The execution time of the benchmark in nanoseconds."""

    @metadata.validator
    def _convert_metadata_to_string(self, _: Any, metadata: dict[Any, Any]) -> None:
        """Convert the metadata to a string representation.

        The function will convert the metadata to a string representation
        to ensure that the metadata can be written to a csv file.
        """
        for key, value in metadata.items():
            sanitized_key = str(key).replace(" ", "_")
            metadata[sanitized_key] = str(value)

    def to_csv(self) -> str:
        """Return the result as a csv string.

        Returns:
            The csv as a string.
        """
        return self.benchmark_result.to_csv(index=False)
