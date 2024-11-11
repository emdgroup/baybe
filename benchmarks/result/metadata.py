"""Benchmark result metadata."""

from datetime import datetime, timedelta

import git
from attrs import define, field
from attrs.validators import instance_of

from baybe import __version__ as baybe_package_version
from baybe.serialization.mixin import SerialMixin


@define(frozen=True)
class ResultMetadata(SerialMixin):
    """The metadata of a benchmark result."""

    start_datetime: datetime = field(validator=instance_of(datetime))
    """The start datetime of the benchmark."""

    duration: timedelta = field(validator=instance_of(timedelta))
    """The time it took to complete the benchmark."""

    commit_hash: str = field(validator=instance_of(str), init=False)
    """The commit hash of the used BayBE code."""

    baybe_version: str = field(default=baybe_package_version, init=False)
    """The used BayBE version."""

    @commit_hash.default
    def _commit_hash_default(self) -> str:
        """Extract the git commit hash."""
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        return sha
