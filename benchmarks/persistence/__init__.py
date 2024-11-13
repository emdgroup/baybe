"""Module for persisting benchmarking results."""

from benchmarks.persistence.persistence import (
    create_persistence_object,
    persister_factory,
)

__all__ = ["persister_factory", "create_persistence_object"]
