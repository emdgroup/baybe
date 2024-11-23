"""Module for persisting benchmarking results."""

from benchmarks.persistence.persistence import (
    make_object_writer,
    make_path_constructor,
)

__all__ = ["make_path_constructor", "make_object_writer"]
