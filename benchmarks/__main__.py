"""Executes the benchmarking module."""

# Run this via 'python -m benchmarks' from the root directory.

import argparse
import os
from collections.abc import Collection
from pathlib import Path

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
    outdir: Path,
    file_name_prefix: str | None = None,
) -> None:
    """Save the benchmark data to the object storage.

    Args:
        benchmark: The benchmark instance that was executed.
        result: The result of the benchmark execution.
        outdir: The directory where the results should be saved.
        file_name_prefix: Additional string that is added to the
                    generated file name.
    """
    path_constructor = PathConstructor.from_result(result)
    persist_dict = benchmark.to_dict() | result.to_dict()

    object_storage = (
        S3ObjectStorage()
        if RUNS_IN_CI
        else LocalFileObjectStorage(
            runmode=result.runmode,
            folder_path_prefix=outdir,
            file_name_prefix=file_name_prefix,
        )
    )
    object_storage.write_json(persist_dict, path_constructor)


def run_benchmarks(
    benchmark_list: Collection[Benchmark],
    runmode: RunMode,
    outdir: Path,
    file_name_prefix: str | None = None,
    save: bool = True,
) -> None:
    """Run a subset based on the benchmark name.

    Args:
        benchmark_list: The list of benchmarks to run.
        runmode: The run mode to use for the benchmarks which decide the used settings.
        outdir: The directory where the results should be saved.
        file_name_prefix: Additional string that is added to the
                            generated file name.
        save: Whether to save the results to the object storage.
    """
    for benchmark in benchmark_list:
        result = benchmark(runmode=runmode)

        if save:
            save_benchmark_data(
                benchmark,
                result,
                file_name_prefix=file_name_prefix,
                outdir=outdir,
            )


def main() -> None:
    """Execute the benchmarking module."""
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
        type=RunMode.from_string,
        help="Runmode setting to use.",
        default=RunMode.DEFAULT,
        choices=list(RunMode),
    )
    parser.add_argument(
        "--file-name-prefix",
        "-p",
        help=(
            "Additional string that is added as a prefix to the"
            " generated file name inside of the folder."
        ),
        default=None,
        type=str,
    )
    parser.add_argument(
        "--outdir",
        "-o",
        help="Save files into directory.",
        default=Path("."),
        type=Path,
    )
    parser.add_argument(
        "-d",
        "--dry-run",
        help=(
            "Run benchmarks without saving results. Note that this will "
            "cause all other storing-related flags to be ignored."
        ),
        action="store_true",
    )
    args = parser.parse_args()

    if args.outdir != Path(".") and RUNS_IN_CI:
        raise ValueError("Output directory cannot be set in CI mode.")

    if args.file_name_prefix and RUNS_IN_CI:
        raise ValueError("File name prefix cannot be set in CI mode.")

    if not args.outdir.exists():
        raise FileNotFoundError(
            f"Output directory {args.outdir} does not exist. "
            "Please create it before running the benchmarks."
        )

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
        file_name_prefix=args.file_name_prefix,
        outdir=args.outdir,
        save=not args.dry_run,
    )


if __name__ == "__main__":
    main()
