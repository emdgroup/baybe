"""Run the benchmarks for the given scenario."""

from benchmark.domain import SINGLE_BENCHMARKS_TO_RUN
from benchmark.result.basic_results import SingleResult


def main():
    """Run the performance test for the given scenario."""
    for benchmark in SINGLE_BENCHMARKS_TO_RUN:
        try:
            result_benchmarking: SingleResult = benchmark.execute_benchmark()
            print(f"Result: {result_benchmarking}")
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
