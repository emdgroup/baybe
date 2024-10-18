"""This module contains the base class for persisting benchmarking results."""

from abc import ABC, abstractmethod
from datetime import datetime
from uuid import UUID

from attr import define, field
from pandas import DataFrame

from baybe import __version__
from src.result import Result


@define
class ResultPersistenceInterface(ABC):
    """Interface for classes that persist experiment results."""

    @staticmethod
    def _default_baybe_version() -> str:
        """Sanitizes the Baybe version by removing any post-release version information.

        Returns:
        - str: The sanitized Baybe version.
        """
        POST_RELEASE_VERSION = len(__version__.split(".")) > 3
        if POST_RELEASE_VERSION:
            return ".".join(__version__.split(".")[:3])
        return __version__

    date_time: datetime
    baybe_version: str = field(factory=_default_baybe_version)
    """The version of the Baybe library."""

    @abstractmethod
    def persist_new_result(self, result: Result) -> None:
        """Store the result of a performance test.

        Args:
            result: The result to be persisted.
        """
        pass

    @abstractmethod
    def load_compare_result(self, experiment_id: UUID) -> DataFrame:
        """Load the oldest stable result for a given experiment ID.

        Loads the oldest result from an experiment that is created from the main branch
        of the Baybe library. This is done to compare the performance of the library
        over a longer time period and to ensure that the results don't just a bit from
        version to version which would be not noticeable in the short term.

        Parameters:
            experiment_id: The ID of the experiment.

        Returns:
            Dataframe: The last result for the given experiment ID.
        """
        pass
