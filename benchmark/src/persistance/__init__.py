"""Module for persisting benchmarking results."""

from benchmark.src.persistance.base import ResultPersistenceInterface
from benchmark.src.persistance.data_handling_classes import (
    LocalExperimentResultPersistence,
    S3ExperimentResultPersistence,
)

__all__ = [
    "ResultPersistenceInterface",
    "S3ExperimentResultPersistence",
    "LocalExperimentResultPersistence",
]
