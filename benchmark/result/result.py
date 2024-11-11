"""Basic result classes for benchmarking."""

from typing import Any
from uuid import UUID

from attrs import define, field
from attrs.validators import instance_of
from pandas import DataFrame

from baybe.serialization.core import converter
from baybe.serialization.mixin import SerialMixin


@define(frozen=True)
class Result(SerialMixin):
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


def _unstructure_uuid_hook(obj: UUID) -> str:
    """Unstructure hook for UUID to string for serialization."""
    return str(obj)


def _structure_uuid_hook(obj: str) -> UUID:
    """Structure hook for string to UUID for deserialization."""
    return UUID(obj)


converter.register_unstructure_hook(UUID, _unstructure_uuid_hook)
converter.register_structure_hook(UUID, _structure_uuid_hook)
