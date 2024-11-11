"""Benchmark result metadata."""

from datetime import datetime, timedelta

import git
from attrs import define, field
from attrs.validators import instance_of
from cattrs.gen import make_dict_unstructure_fn

from baybe import __version__ as baybe_package_version
from baybe.serialization.core import converter
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

    last_published_baybe_version: str = field(init=False)
    """The used BayBE version."""

    @commit_hash.default
    def _commit_hash_default(self) -> str:
        """Extract the git commit hash."""
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        return sha

    @last_published_baybe_version.default
    def _baybe_version_default(self) -> str:
        """Extract the BayBE version."""
        POST_SUBVERSION_CONSTRUCTED = baybe_package_version.count(".") > 2
        if POST_SUBVERSION_CONSTRUCTED:
            CORE_BAYBE_VERSION = baybe_package_version[
                : baybe_package_version.rfind(".")
            ]
            return CORE_BAYBE_VERSION
        return baybe_package_version


# Register un-/structure hooks
converter.register_unstructure_hook(
    ResultMetadata,
    make_dict_unstructure_fn(
        ResultMetadata, converter, _cattrs_include_init_false=True
    ),
)
