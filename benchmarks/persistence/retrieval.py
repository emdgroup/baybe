"""Classes for fetching benchmark input data."""

from __future__ import annotations

import os
from functools import cached_property
from pathlib import Path
from typing import Protocol, runtime_checkable

import boto3
import boto3.session
from attr import define, field
from boto3.session import Session
from typing_extensions import override

VARNAME_BENCHMARKING_DATA_BUCKET = "BAYBE_BENCHMARKING_DATA_BUCKET"


@runtime_checkable
class ObjectFetcherProtocol(Protocol):
    """Interface for fetching a single benchmark input file by key."""

    __slots__ = ()

    def fetch(self, key: str, destination: Path) -> None:
        """Download the object identified by key to destination.

        Args:
            key: The identifier of the object at the fetcher's backend.
            destination: The local path to download the object to.
        """


@define
class CachedFileLoader:
    """Loads benchmark input files, fetching them once if not cached locally."""

    fetcher: ObjectFetcherProtocol = field()
    """The backend used to fetch files that are not yet cached locally."""

    def get_file(self, key: str, local_path: Path) -> Path:
        """Return the local path to a file, fetching it if not already cached.

        Args:
            key: The identifier of the file at the fetcher's backend.
            local_path: The local path at which the file is cached.

        Returns:
            The local path to the file.
        """
        if not local_path.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self.fetcher.fetch(key, local_path)
        return local_path


@define(slots=False)
class S3ObjectRetrieval(ObjectFetcherProtocol):
    """Class for fetching benchmark input files from an S3 bucket.

    Counterpart to :class:`~benchmarks.persistence.persistence.S3ObjectStorage` for
    the input side: instead of writing results, it fetches input files from S3 by
    key. Used as the fetcher backend of a :class:`CachedFileLoader`, which adds local
    caching.

    The S3 bucket name is resolved lazily on first use (not at construction time),
    so that instantiating this class does not require S3 access to be configured
    when the requested files are already available in the local cache.
    """

    _object_session: Session = field(factory=boto3.session.Session)
    """The boto3 session object. Loads the required credentials
    from environment variables."""

    @cached_property
    def _bucket_name(self) -> str:
        """The name of the S3 bucket from which input data is fetched."""
        if VARNAME_BENCHMARKING_DATA_BUCKET not in os.environ:
            raise ValueError(
                f"No S3 bucket name provided for benchmark input data. Please "
                f"provide the bucket name by setting the environment variable "
                f"'{VARNAME_BENCHMARKING_DATA_BUCKET}'."
            )
        return os.environ[VARNAME_BENCHMARKING_DATA_BUCKET]

    @override
    def fetch(self, key: str, destination: Path) -> None:
        """Download an object from the S3 bucket to a local destination.

        Args:
            key: The S3 key of the object.
            destination: The local path to download the object to.
        """
        client = self._object_session.client("s3")
        client.download_file(self._bucket_name, key, str(destination))
