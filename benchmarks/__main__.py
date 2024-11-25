"""Executes the benchmarking module."""
# Run this via 'python -m benchmarks' from the root directory.

import os

from benchmarks.domains import BENCHMARKS
from benchmarks.persistence import (
    LocalFileObjectStorage,
    PathConstructor,
    S3ObjectStorage,
)

RUNS_IN_CI = "CI" in os.environ


def main():
    """Run all benchmarks."""
    for benchmark in BENCHMARKS:
        result = benchmark()
        path_constructor = PathConstructor.from_result(result)
        persist_dict = benchmark.to_dict() | result.to_dict()

        object_storage = S3ObjectStorage() if RUNS_IN_CI else LocalFileObjectStorage()
        object_storage.write_json(persist_dict, path_constructor)


if __name__ == "__main__":
    main()
