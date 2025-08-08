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


def save_benchmark_data(
    benchmark: Benchmark,
    result: Result,
    name: str | None = None,
    outdir: str | None = None,
) -> None:
    """Save the benchmark data to the object storage."""
    path_constructor = PathConstructor.from_result(result, name=name, outdir=outdir)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    persist_dict = benchmark.to_dict() | result.to_dict()

    object_storage = S3ObjectStorage() if RUNS_IN_CI else LocalFileObjectStorage()
    object_storage.write_json(persist_dict, path_constructor)


def run_all_benchmarks(
    smoketest: str | None = None,
    name: str | None = None,
    outdir: str | None = None,
) -> None:
    """Run all benchmarks."""
    for benchmark in BENCHMARKS:
        result = benchmark(smoketest=smoketest)
        save_benchmark_data(benchmark, result, name=name, outdir=outdir)


def run_benchmarks(
    benchmark_names: Collection[str],
    smoketest: str | None = None,
    name: str | None = None,
    outdir: str | None = None,
) -> None:
    """Run a subset based on the benchmark name."""
    for benchmark in BENCHMARKS:
        if benchmark.name not in benchmark_names:
            continue

        result = benchmark(smoketest=smoketest)
        save_benchmark_data(benchmark, result, name=name, outdir=outdir)


def main() -> None:
    """Execute the benchmarking module."""
    parser = argparse.ArgumentParser(description="Executes the benchmarking module.")
    parser.add_argument(
        "--benchmark-list", nargs="+", help="List of benchmarks to run", default=None
    )
    parser.add_argument(
        "--smoketest", type=str, help="Smoketest setting to use.", default=None
    )
    parser.add_argument(
        "--name", help="Additional name to add to saved file.", default=None
    )
    parser.add_argument("--outdir", help="Save files into directory.", default=None)
    args = parser.parse_args()
    if not args.benchmark_list:
        run_all_benchmarks(smoketest=args.smoketest, name=args.name)
        return

    benchmark_execute_set = set(args.benchmark_list)
    run_benchmarks(
        benchmark_execute_set,
        smoketest=args.smoketest,
        name=args.name,
        outdir=args.outdir,
    )


if __name__ == "__main__":
    main()
