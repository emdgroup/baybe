"""Basic result classes for benchmarking."""

from typing import Any
from uuid import UUID

from attrs import define, field
from attrs.validators import instance_of
from pandas import DataFrame


@define(frozen=True)
class Result:
    """A single result of the benchmarking."""

    benchmark_origin_id: UUID = field(validator=instance_of(UUID))
    """The unique identifier of the benchmark running which can be set
    to compare different executions of the same benchmark setting."""

    metadata: dict[str, Any] = field(validator=instance_of(dict))
    """Metadata about the benchmarking result."""

    benchmark_result: DataFrame = field()
    """The result of the benchmarked callable."""

    execution_time_sec: float = field(validator=instance_of(float))
    """The execution time of the benchmark in seconds."""

    @metadata.validator
    def _validate_metadata(self, _: Any, value: dict[str, Any]) -> None:
        if not isinstance(value, dict):
            raise ValueError(f"Metadata must be a dictionary, got {type(value)}")
        for key in value.keys():
            if not isinstance(key, str):
                raise ValueError(f"Metadata keys must be strings, got {type(key)}")
