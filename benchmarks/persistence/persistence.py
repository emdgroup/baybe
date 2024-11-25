"""Classes for persisting benchmark results."""

from __future__ import annotations

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
from typing_extensions import override

from benchmarks import Result

VARNAME_BENCHMARKING_PERSISTENCE_PATH = "BAYBE_BENCHMARKING_PERSISTENCE_PATH"


class PathStrategy(Enum):
    """Specifies the way a file path is constructed."""

    HIERARCHICAL = "HIERARCHICAL"
    """Hierarchical path construction using folders."""

    FLAT = "FLAT"
    """Flat path construction using a file name only."""


@define
class PathConstructor:
    """A class to construct the file path of a result object.

    The class encapsulates the construction of a file path depending
    on where the object should be stored. Since different storage backends have
    different requirements for the path, and some like the used S3 bucket
    do not support folders, this class uses its variables
    to construct a `Path` object depending on the strategy.

    For compatibility reasons, the path is sanitized to contain only
    lower and upper case letters, digits as well as the symbols '.' and '-'"
    """

    benchmark_name: str = field(validator=instance_of(str))
    """The name of the benchmark for which the path should be constructed."""

    branch: str = field(validator=instance_of(str))
    """The branch checked out at benchmark execution time."""

    latest_baybe_tag: str = field(validator=instance_of(str))
    """The latest BayBE version tag existing at benchmark execution time."""

    execution_date_time: datetime = field(validator=instance_of(datetime))
    """The date and time when the benchmark was executed."""

    commit_hash: str = field(validator=instance_of(str))
    """The hash of the commit checked out at benchmark execution time."""

    def _sanitize_string(self, string: str) -> str:
        """Replace disallowed characters for filesystems in the given string."""
        ALLOWED_CHARS = (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-."
        )
        return "".join([char if char in ALLOWED_CHARS else "-" for char in string])

    @classmethod
    def from_result(cls, result: Result) -> PathConstructor:
        """Create a path constructor from result.

        Args:
            result: The result of the benchmark.

        Returns:
            The path constructor.
        """
        benchmark_name = result.benchmark_identifier
        start_datetime = result.metadata.start_datetime
        commit_hash = result.metadata.commit_hash
        latest_baybe_tag = result.metadata.latest_baybe_tag
        branch = result.metadata.branch

        return PathConstructor(
            benchmark_name=benchmark_name,
            latest_baybe_tag=latest_baybe_tag,
            commit_hash=commit_hash,
            execution_date_time=start_datetime,
            branch=branch,
        )

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
            self._sanitize_string(component) for component in components
        ]
        path = separator.join(sanitized_components) + separator + "result.json"

        return Path(path)


class ObjectStorage(Protocol):
    """Interface for interacting with storage."""

    def write_json(self, object: dict, path_constructor: PathConstructor) -> None:
        """Store a JSON serializable dictionary according to the path_constructor.

        If the respective object exists, it will be overwritten.

        Args:
            object: The object to be persisted.
            path_constructor: The path constructor creating the path for the object.
        """


@define
class S3ObjectStorage(ObjectStorage):
    """Class for persisting objects in an S3 bucket."""

    _bucket_name: str = field(validator=instance_of(str), init=False)
    """The name of the S3 bucket where the objects are stored."""

    _object_session: Session = field(factory=boto3.session.Session)
    """The boto3 session object. Loads the required credentials
    from environment variables."""

    @_bucket_name.default
    def _default_bucket_name(self) -> str:
        """Get the bucket name from the environment variables."""
        if VARNAME_BENCHMARKING_PERSISTENCE_PATH not in os.environ:
            raise ValueError(
                f"No S3 bucket name provided. Please provide the "
                f"bucket name by setting the environment variable "
                f"'{VARNAME_BENCHMARKING_PERSISTENCE_PATH}'."
            )
        return os.environ[VARNAME_BENCHMARKING_PERSISTENCE_PATH]

    @override
    def write_json(self, object: dict, path_constructor: PathConstructor) -> None:
        """Store a JSON serializable dictionary in an S3 bucket.

        The S3-key of the JSON is created from
        the path_constructor. If the key already exists, it will be overwritten.

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
class LocalFileObjectStorage(ObjectStorage):
    """Class for persisting objects locally."""

    folder_path_prefix: Path = field(converter=Path, default=Path("."))
    """The prefix of the folder path where the results are stored."""

    @folder_path_prefix.validator
    def _folder_path_prefix_validator(self, _, folder_path_prefix: Path) -> None:
        """Validate the existence of the path."""
        if not folder_path_prefix.exists():
            raise FileNotFoundError(
                f"The folder path '{folder_path_prefix.resolve()}' does not exist."
            )

    @override
    def write_json(self, object: dict, path_constructor: PathConstructor) -> None:
        """Store a JSON serializable dictionary in the local file system.

        If the respective file exists, it will be overwritten.

        Args:
            object: The object to be persisted.
            path_constructor: The path constructor creating the path for the object.
        """
        path_object = self.folder_path_prefix.joinpath(
            path_constructor.get_path(strategy=PathStrategy.FLAT)
        )
        with open(path_object.resolve(), "w") as file:
            json.dump(object, file)
