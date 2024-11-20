"""Classes for persisting benchmark results."""

import json
import os
from pathlib import Path
from typing import Protocol

import boto3
import boto3.session
from attr import define, field
from attrs.validators import instance_of
from git import Repo

from benchmarks import Benchmark, Result

MODULE_IS_RUNNING_IN_THE_PIPELINE = "GITHUB_RUN_ID" in os.environ


class PersistenceObjectInterface(Protocol):
    """Interface for constructing the path of a result object."""

    def get_full_file_path(self) -> str:
        """Construct the path of a result object.

        Returns:
            The path of the result object.
        """


class PersistenceHandlingInterface(Protocol):
    """Interface for persisting experiment results."""

    def write_object(self, path: PersistenceObjectInterface, object: dict) -> None:
        """Store a JSON serializable dict according to the path.

        Args:
            path: The path of the object to be persisted
            object: The object to be persisted.
        """


@define
class S3PersistenceObjectCreator:
    """Class for creating the path of a result object."""

    _benchmark: Benchmark = field(validator=instance_of(Benchmark))
    """The benchmark for which the result is stored."""

    _result: Result = field(validator=instance_of(Result))
    """The result of the benchmark."""

    _branch: str = field(validator=instance_of(str), init=False)
    """The branch of the BayBE library from which the workflow was started."""

    _workflow_run_identifier: str = field(validator=instance_of(str), init=False)
    """The identifier of the workflow run."""

    @_branch.default
    def _default_branch(self) -> str:
        repo = Repo(search_parent_directories=True)
        current_branch = repo.active_branch.name
        S3_PATH_FORBIDDEN_CHARACTERS = ["/", "\\"]
        sanitized_branch = ""
        for char in S3_PATH_FORBIDDEN_CHARACTERS:
            sanitized_branch = current_branch.replace(char, "-")
        return sanitized_branch

    @_workflow_run_identifier.default
    def _default_workflow_id(self) -> str:
        """Get the workflow run identifier."""
        if MODULE_IS_RUNNING_IN_THE_PIPELINE:
            return os.environ["GITHUB_RUN_ID"]
        raise ValueError("The environment variable GITHUB_RUN_ID is not set.")

    def get_full_file_path(self) -> str:
        """Construct the path of a result object.

        Returns:
            The path of the result object.
        """
        experiment_identifier = self._benchmark.name
        metadata = self._result.metadata
        file_usable_date = metadata.start_datetime.strftime("%Y-%m-%d")
        bucket_path_key = (
            f"{experiment_identifier}/{self._branch}/{metadata.latest_baybe_tag}/"
            + f"{file_usable_date}/"
            + f"{metadata.commit_hash}/{self._workflow_run_identifier}/"
            + "result.json"
        )
        return bucket_path_key


@define
class S3PersistenceHandler:
    """Class for persisting experiment results in an S3 bucket."""

    _bucket_name: str = field(validator=instance_of(str), init=False)
    """The name of the S3 bucket where the results are stored."""

    _object_session = boto3.session.Session()
    """The boto3 session object. This will load the respective credentials
    from the environment variables within the container."""

    def write_object(self, path: PersistenceObjectInterface, object: dict) -> None:
        """Store a JSON serializable dict according to the path.

        This method will store an JSON serializable dict in an S3 bucket.
        The S3-key of the Java Script Notation Object will be the experiment identifier,
        the branch, the BayBE-version, the start datetime, the commit hash and the
        workflow run identifier.

        Args:
            path: The path of the object to be persisted
            object: The object to be persisted.
        """
        client = self._object_session.client("s3")

        client.put_object(
            Bucket=self._bucket_name,
            Key=path.get_full_file_path(),
            Body=json.dumps(object),
            ContentType="application/json",
        )


@define
class LocalFileSystemObjectCreator:
    """Class for creating the path of a result object."""

    _benchmark: Benchmark = field(validator=instance_of(Benchmark))
    """The benchmark for which the result is stored."""

    _result: Result = field(validator=instance_of(Result))
    """The result of the benchmark."""

    _path: str = field(validator=instance_of(str))
    """The path where the result is stored."""

    _filename: str = field(validator=instance_of(str))
    """The filename of the result object."""

    @_path.default
    def _default_bucket_name(self) -> str:
        ENVIRONMENT_NOT_SET = "BAYBE_BENCHMARK_PERSISTENCE_PATH" not in os.environ
        if ENVIRONMENT_NOT_SET:
            return "./"
        return os.environ["BAYBE_BENCHMARK_PERSISTENCE_PATH"]

    @_filename.default
    def _default_filename(self) -> str:
        experiment_identifier = self._benchmark.name
        metadata = self._result.metadata
        file_usable_date = metadata.start_datetime.strftime("%Y-%m-%d")
        return (
            f"{experiment_identifier}_{metadata.latest_baybe_tag}"
            + f"_{file_usable_date}_{metadata.commit_hash}.json"
        )

    def get_full_file_path(self) -> str:
        """Construct the path of a result object.

        Returns:
            The path of the result object.
        """
        return Path(f"{self._path}/{self._filename}").resolve().as_posix()


@define
class LocalFileSystemPersistenceHandler:
    """Class for persisting experiment results in an S3 bucket."""

    def write_object(self, path: PersistenceObjectInterface, object: dict) -> None:
        """Store a JSON serializable dict according to the path.

        This method will store an JSON serializable dict in a local file system.

        Args:
            path: The path of the object to be persisted
            object: The object to be persisted.
        """
        with open(path.get_full_file_path(), "w") as file:
            json.dump(object, file)


def persister_factory() -> PersistenceHandlingInterface:
    """Create a persistence handler based on the environment variables.

    Returns:
        The persistence handler.
    """
    if MODULE_IS_RUNNING_IN_THE_PIPELINE:
        return S3PersistenceHandler()
    return LocalFileSystemPersistenceHandler()


def persistence_object_factory(
    benchmark: Benchmark, result: Result
) -> PersistenceObjectInterface:
    """Create a persistence object.

    Args:
        benchmark: The benchmark for which the result is stored.
        result: The result of the benchmark.

    Returns:
        The persistence object.
    """
    if MODULE_IS_RUNNING_IN_THE_PIPELINE:
        return S3PersistenceObjectCreator(benchmark, result)
    return LocalFileSystemObjectCreator(benchmark, result)
