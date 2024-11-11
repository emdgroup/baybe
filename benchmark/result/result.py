"""Basic result classes for benchmarking."""

from uuid import UUID

from attrs import define, field
from attrs.validators import instance_of
from pandas import DataFrame

from benchmark.result.utils import _convert_metadata_to_string


@define(frozen=True)
class Result:
    """A single result of the benchmarking."""

    benchmark_origin_id: UUID = field(validator=instance_of(UUID))
    """The unique identifier of the benchmark running which can be set
    to compare different executions of the same benchmark setting."""

    metadata: dict[str, str] = field(
        validator=instance_of(dict), converter=_convert_metadata_to_string
    )
    """Metadata about the benchmarking result."""

    benchmark_result: DataFrame = field()
    """The result of the benchmarked callable."""

    execution_time_ns: int = field(validator=instance_of(int))
    """The execution time of the benchmark in nanoseconds."""

    def to_csv(self) -> str:
        """Return the result as a csv string.

        Returns:
            The csv as a string.
        """
        return self.benchmark_result.to_csv(index=False)
