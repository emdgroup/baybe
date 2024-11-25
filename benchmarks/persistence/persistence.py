"""Classes for persisting benchmark results."""

import json
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Protocol

import boto3
import boto3.session
from attr import define, field
from attrs.validators import instance_of
from boto3.session import Session
from git import Repo
from typing_extensions import override

from benchmarks import Benchmark, Result

VARNAME_BENCHMARKING_PERSISTENCE_PATH = "BAYBE_BENCHMARKING_PERSISTENCE_PATH"
PERSIST_DATA_TO_S3_BUCKET = VARNAME_BENCHMARKING_PERSISTENCE_PATH in os.environ

VARNAME_GITHUB_CI = "CI"
RUNS_ON_GITHUB_CI = VARNAME_GITHUB_CI in os.environ


class PathStrategy(Enum):
    """The way a path extension is constructed."""

    HIERARCHICAL = "HIERARCHICAL"
    """Hierarchical path construction using folders."""

    FLAT = "FLAT"
    """Flat path construction using a file name only."""


@define
class PathConstructor:
    """A class to construct the hierarchical path of a result object.

    This class is used to encapsulate the construction of a respective path depending
    on where the object should be stored. Since different storage backends have
    different requirements for the path, and some like the used S3 Bucket
    do not support folders, the path uses its variables to construct the path
    in the order of the variables with a separator depending on the strategy.
    For compatibility reasons, the path is sanitized to only contain the following
    allowed characters:
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-."
    """

    benchmark_name: str = field(validator=instance_of(str))
    """The name of the benchmark for which the path should be constructed."""

    branch: str = field(validator=instance_of(str), init=False)
    """The branch currently checked out."""

    latest_baybe_tag: str = field(validator=instance_of(str))
    """The latest BayBE version tag."""

    execution_date_time: datetime = field(validator=instance_of(datetime))
    """The date and time when the benchmark was executed."""

    commit_hash: str = field(validator=instance_of(str))
    """The hash of the commit currently checked out."""

    @branch.default
    def _default_branch(self) -> str:
        repo = Repo(search_parent_directories=True)
        current_branch = repo.active_branch.name
        return current_branch

    def _string_sanitizer(self, string: str) -> str:
        """Replace disallowed characters for filesystems in the given string."""
        ALLOWED_CHARS = (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-."
        )
        return "".join([char if char in ALLOWED_CHARS else "-" for char in string])

    def get_path(self, strategy: PathStrategy) -> Path:
        """Construct the path of a result object.

        Creates the path depending on the chosen strategy by concatenating
        the experiment identifier, the branch, the BayBE version,
        the start date (without time information) and the commit hash.

        Args:
            strategy: The strategy to construct the path.

        Returns:
            The path to persist the object. Can be a file name or
            a folder path as a string.
        """
        separator = "/" if strategy is PathStrategy.HIERARCHICAL else "_"

        file_usable_date = self.execution_date_time.strftime("%Y-%m-%d")
        components = [
            self.benchmark_name,
            self.branch,
            self.latest_baybe_tag,
            file_usable_date,
            self.commit_hash,
        ]

        sanitized_components = [
            self._string_sanitizer(component) for component in components
        ]
        path = separator.join(sanitized_components) + separator + "result.json"

        return Path(path)


class ObjectStorage(Protocol):
    """Interface for interacting with storage."""

    def write_json(self, object: dict, path_constructor: PathConstructor) -> None:
        """Store a JSON serializable dict according to the path.

        If the respective object exists, it will be overwritten.

        Args:
            object: The object to be persisted.
            path_constructor: The path constructor creating the path for the object.
        """


@define
class S3ObjectStorage(ObjectStorage):
    """Class for persisting objects in an S3 bucket."""

    _bucket_name: str = field(validator=instance_of(str), init=False)
    """The name of the S3 bucket where the results are stored."""

    _object_session: Session = field(factory=boto3.session.Session)
    """The boto3 session object. This will load the respective credentials
    from the environment variables within the container."""

    @_bucket_name.default
    def _default_bucket_name(self) -> str:
        """Get the bucket name from the environment variables."""
        if not PERSIST_DATA_TO_S3_BUCKET:
            raise ValueError(
                f"No S3 bucket name provided. Please provide the "
                f"bucket name by setting the environment variable "
                f"'{VARNAME_BENCHMARKING_PERSISTENCE_PATH}'."
            )
        return os.environ[VARNAME_BENCHMARKING_PERSISTENCE_PATH]

    @override
    def write_json(self, object: dict, path_constructor: PathConstructor) -> None:
        """Store a JSON serializable dict according to the path.

        This method will store an JSON serializable dict in an S3 bucket.
        The S3-key of the Java Script Notation Object will be created by
        the path object. I the key already exists, it will be overwritten.

        Args:
            object: The object to be persisted.
            path_constructor: The path constructor creating the path for the object.

        """
        client = self._object_session.client("s3")

        key = path_constructor.get_path(strategy=PathStrategy.HIERARCHICAL)

        client.put_object(
            Bucket=self._bucket_name,
            Key=key.as_posix(),
            Body=json.dumps(object),
            ContentType="application/json",
        )


@define
class LocalFileSystemObjectStorage(ObjectStorage):
    """Class for persisting JSON serializable dicts locally."""

    folder_path_prefix: Path = field(converter=Path, default=Path("."))
    """The prefix of the folder path where the results are stored.
    The filename will be created automatically and create or override the
    file under this path. The file path must exist."""

    @override
    def write_json(self, object: dict, path_constructor: PathConstructor) -> None:
        """Store a JSON serializable dict according to the path.

        This method will store an JSON serializable dict in a local file system.
        If the respective file exists, it will be overwritten.

        Args:
            object: The object to be persisted.
            path_constructor: The path constructor creating the path for the object.


        Raises:
            FileNotFoundError: If the folder path prefix does not exist.
        """
        if not self.folder_path_prefix.exists():
            raise FileNotFoundError(
                f"The folder path {self.folder_path_prefix.resolve()} does not exist."
            )
        path_object = self.folder_path_prefix.joinpath(
            path_constructor.get_path(strategy=PathStrategy.FLAT)
        )
        with open(path_object.resolve(), "w") as file:
            json.dump(object, file)


def make_object_writer() -> ObjectStorage:
    """Create a persistence handler based on the environment variables.

    Returns:
        The persistence handler.
    """
    if not RUNS_ON_GITHUB_CI:
        return LocalFileSystemObjectStorage()
    return S3ObjectStorage()


def make_path_constructor(benchmark: Benchmark, result: Result) -> PathConstructor:
    """Create a path constructor.

    Args:
        benchmark: The benchmark for which the result is stored.
        result: The result of the benchmark.

    Returns:
        The persistence path constructor.
    """
    benchmark_name = benchmark.name
    start_datetime = result.metadata.start_datetime
    commit_hash = result.metadata.commit_hash
    latest_baybe_tag = result.metadata.latest_baybe_tag

    return PathConstructor(
        benchmark_name=benchmark_name,
        latest_baybe_tag=latest_baybe_tag,
        commit_hash=commit_hash,
        execution_date_time=start_datetime,
    )
