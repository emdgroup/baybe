"""Executes the benchmarking module."""
# Run this via 'python -m benchmarks' from the root directory.

import argparse
import os
from collections.abc import Collection

from benchmarks.definition import Benchmark
from benchmarks.domains import BENCHMARKS
from benchmarks.persistence import (
    LocalFileObjectStorage,
    PathConstructor,
    S3ObjectStorage,
)
from benchmarks.result import Result

RUNS_IN_CI = "CI" in os.environ


def save_benchmark_data(benchmark: Benchmark, result: Result) -> None:
    """Save the benchmark data to the object storage."""
    path_constructor = PathConstructor.from_result(result)
    persist_dict = benchmark.to_dict() | result.to_dict()

    object_storage = S3ObjectStorage() if RUNS_IN_CI else LocalFileObjectStorage()
    object_storage.write_json(persist_dict, path_constructor)


def run_all_benchmarks() -> None:
    """Run all benchmarks."""
    for benchmark in BENCHMARKS:
        result = benchmark()
        save_benchmark_data(benchmark, result)


def run_benchmarks(benchmark_names: Collection[str]) -> None:
    """Run a subset based on the benchmark name."""
    for benchmark in BENCHMARKS:
        if benchmark.name not in benchmark_names:
            continue

        result = benchmark()
        save_benchmark_data(benchmark, result)


def main() -> None:
    """Execute the benchmarking module."""
    parser = argparse.ArgumentParser(description="Executes the benchmarking module.")
    parser.add_argument(
        "--benchmark-list", nargs="+", help="List of benchmarks to run", default=None
    )
    args = parser.parse_args()
    if not args.benchmark_list:
        run_all_benchmarks()
        return

    benchmark_execute_set = set(args.benchmark_list)
    run_benchmarks(benchmark_execute_set)


if __name__ == "__main__":
    main()
