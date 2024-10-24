"""Run the benchmarks for the given scenario."""

import logging

from benchmark.domain import BENCHMARKS
from benchmark.result.result import Result


def main():
    """Run the performance test for the given scenario."""
    for benchmark in BENCHMARKS:
        result_benchmarking: Result = benchmark.run()
        logging.info(f"Result: {result_benchmarking}")


if __name__ == "__main__":
    main()
