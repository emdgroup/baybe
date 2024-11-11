"""This module contains the metadata of a benchmark result."""

from datetime import datetime

import git
from attrs import define, field
from attrs.validators import instance_of

from baybe import __version__ as baybe_version
from baybe.serialization.mixin import SerialMixin


@define(frozen=True)
class ResultMetadata(SerialMixin):
    """The metadata of a benchmark result."""

    benchmark_name: str = field(validator=instance_of(str))
    """The name of the benchmark."""

    execution_time_sec: float = field(validator=instance_of(float))
    """The execution time of the benchmark in seconds."""

    gpu_used: bool = field(validator=instance_of(bool))
    """Whether the benchmark used a GPU."""

    start_datetime: str = field(
        validator=instance_of(str), factory=datetime.now().isoformat
    )
    """The start datetime of the benchmark."""

    @property
    def baybe_version(self) -> str:
        """The version of the baybe package."""
        return baybe_version

    @property
    def commit_hash(self) -> str:
        """The commit hash of the baybe package."""
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        return sha
