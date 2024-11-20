"""Module for persisting benchmarking results."""

from benchmarks.persistence.persistence import (
    persistence_object_factory,
    persister_factory,
)

__all__ = ["persister_factory", "persistence_object_factory"]
