"""Benchmark result metadata."""

from datetime import datetime, timedelta

import git
from attrs import define, field
from attrs.validators import instance_of
from cattrs.gen import make_dict_unstructure_fn

from benchmarks.serialization import BenchmarkSerialization, converter


@define(frozen=True)
class ResultMetadata(BenchmarkSerialization):
    """The metadata of a benchmark result."""

    start_datetime: datetime = field(validator=instance_of(datetime))
    """The start datetime of the benchmark."""

    duration: timedelta = field(validator=instance_of(timedelta))
    """The time it took to complete the benchmark."""

    commit_hash: str = field(validator=instance_of(str), init=False)
    """The commit hash of the used BayBE code."""

    latest_baybe_tag: str = field(validator=instance_of(str), init=False)
    """The latest BayBE tag reachable in the ancestor commit history."""

    branch: str = field(validator=instance_of(str), init=False)
    """The branch currently checked out."""

    @branch.default
    def _default_branch(self) -> str:
        """Set the current checkout branch."""
        repo = git.Repo(search_parent_directories=True)
        current_branch = repo.active_branch.name
        return current_branch

    @commit_hash.default
    def _default_commit_hash(self) -> str:
        """Extract the git commit hash."""
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        return sha

    @latest_baybe_tag.default
    def _default_latest_baybe_tag(self) -> str:
        """Extract the latest reachable BayBE tag."""
        repo = git.Repo(search_parent_directories=True)
        latest_tag = repo.git.describe(tags=True, abbrev=0)
        return latest_tag


# Register un-/structure hooks
converter.register_unstructure_hook(
    ResultMetadata,
    make_dict_unstructure_fn(
        ResultMetadata, converter, _cattrs_include_init_false=True
    ),
)
