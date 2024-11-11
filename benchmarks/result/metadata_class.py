"""This module contains the metadata of a benchmark result."""

from datetime import datetime

import git
from attrs import define, field
from attrs.validators import instance_of

from baybe import __version__ as baybe_package_version
from baybe.serialization.core import converter
from baybe.serialization.mixin import SerialMixin


@define(frozen=True)
class ResultMetadata(SerialMixin):
    """The metadata of a benchmark result."""

    execution_time_sec: float = field(validator=instance_of(float))
    """The execution time of the benchmark in seconds."""

    start_datetime: datetime = field(validator=instance_of(datetime))
    """The start datetime of the benchmark."""

    commit_hash: str = field(validator=instance_of(str), init=False)
    """Current commit hash of the baybe package."""

    baybe_version: str = field(default=baybe_package_version, init=False)
    """The version of the baybe package."""

    @commit_hash.default
    def _commit_hash_default(self) -> str:
        """Commit hash of the baybe package."""
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        return sha


converter.register_unstructure_hook(datetime, lambda x: x.isoformat())
converter.register_structure_hook(datetime, lambda x, _: datetime.fromisoformat(x))
