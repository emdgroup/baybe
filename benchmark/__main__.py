"""Run the benchmarks for the given scenario."""

import logging

from benchmark.domain import BENCHMARKS
from benchmark.result.result import Result


def main():
    """Run the performance test for the given scenario."""
    for benchmark in BENCHMARKS:
        benchmark_result: Result = benchmark.run()
        logging.info(
            f"Benchmark {benchmark_result.name}"
            + f"({benchmark_result.identifier}) finished."
        )


if __name__ == "__main__":
    main()
