"""Basic result classes for benchmarking."""

from __future__ import annotations

from uuid import UUID

from attrs import define, field
from attrs.validators import instance_of
from pandas import DataFrame

from baybe.serialization.mixin import SerialMixin
from benchmarks.result.metadata import ResultMetadata


@define(frozen=True)
class Result(SerialMixin):
    """A single result of the benchmarking."""

    benchmark_identifier: UUID = field(validator=instance_of(UUID))
    """The identifier of the benchmark that produced the result."""

    result: DataFrame = field(validator=instance_of(DataFrame))
    """The result of the benchmarked callable."""

    metadata: ResultMetadata = field(validator=instance_of(ResultMetadata))
    """The metadata associated with the benchmark result."""
