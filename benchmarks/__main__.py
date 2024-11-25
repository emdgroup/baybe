"""Executes the benchmarking module."""
# Run this via 'python -m benchmarks' from the root directory.

import os

from benchmarks.domains import BENCHMARKS
from benchmarks.persistence import (
    LocalFileObjectStorage,
    PathConstructor,
    S3ObjectStorage,
)

VARNAME_GITHUB_CI = "CI"
RUNS_ON_GITHUB_CI = VARNAME_GITHUB_CI in os.environ


def main():
    """Run all benchmarks."""
    for benchmark in BENCHMARKS:
        result = benchmark()
        path_constructor = PathConstructor.from_benchmark_and_result(benchmark, result)
        persist_dict = benchmark.to_dict() | result.to_dict()

        if not RUNS_ON_GITHUB_CI:
            object_storage = LocalFileObjectStorage()
        else:
            object_storage = S3ObjectStorage()

        object_storage.write_json(persist_dict, path_constructor)


if __name__ == "__main__":
    main()
