"""Module for persisting benchmarking results."""

from benchmark.persistance.base import ResultPersistenceInterface
from benchmark.persistance.data_handling_classes import (
    LocalExperimentResultPersistence,
    S3ExperimentResultPersistence,
)

__all__ = [
    "ResultPersistenceInterface",
    "S3ExperimentResultPersistence",
    "LocalExperimentResultPersistence",
]
