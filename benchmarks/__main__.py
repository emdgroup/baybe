"""Executes the benchmarking module."""
# Run this via 'python -m benchmarks' from the root directory.

import argparse
import os
from collections.abc import Collection

from benchmarks.definition import Benchmark, RunMode
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
    # TODO should we do something with runmode here as well?
    path_constructor = PathConstructor.from_result(result, name=name, outdir=outdir)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    persist_dict = benchmark.to_dict() | result.to_dict()

    object_storage = S3ObjectStorage() if RUNS_IN_CI else LocalFileObjectStorage()
    object_storage.write_json(persist_dict, path_constructor)


def run_benchmarks(
    benchmark_list: Collection[Benchmark],
    runmode: RunMode,
    name: str | None = None,
    outdir: str | None = None,
    save: bool = True,
) -> None:
    """Run a subset based on the benchmark name."""
    for benchmark in benchmark_list:
        result = benchmark(runmode=runmode)

        if save:
            save_benchmark_data(benchmark, result, name=name, outdir=outdir)


def main() -> None:
    """Execute the benchmarking module."""

    def intstr_to_bool(x):
        return bool(int(x))

    parser = argparse.ArgumentParser(description="Executes the benchmarking module.")
    parser.add_argument(
        "--benchmark-list",
        "-b",
        nargs="+",
        help="List of benchmarks to run, e.g. --benchmark-list direct_arylation"
        " temperature_tl. Runs all benchmarks if not specified.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--runmode",
        "-r",
        type=RunMode,
        help="Runmode setting to use.",
        default=RunMode.STANDARD,
        choices=list(RunMode),
    )
    parser.add_argument(
        "--name",
        "-n",
        help="Additional name to add to saved file.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--outdir", "-o", help="Save files into directory.", default=None, type=str
    )
    parser.add_argument(
        "-s",
        "--save",
        help="Whether to save the benchmark results. Use 1 for True and 0 for False.",
        default=True,
        type=intstr_to_bool,
    )
    args = parser.parse_args()

    if args.outdir and RUNS_IN_CI:
        raise AttributeError("Output directory cannot be set in CI mode.")

    if args.name and RUNS_IN_CI:
        raise AttributeError("Name cannot be set in CI mode.")

    if not args.benchmark_list:
        benchmark_list = BENCHMARKS
    else:
        benchmark_list = [
            benchmark
            for benchmark in BENCHMARKS
            if benchmark.name in set(args.benchmark_list)
        ]

    run_benchmarks(
        benchmark_list=benchmark_list,
        runmode=args.runmode,
        name=args.name,
        outdir=args.outdir,
        save=args.save,
    )


if __name__ == "__main__":
    main()
