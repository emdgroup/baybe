"""Run the benchmarks."""

from benchmark.domains import BENCHMARKS


def main():
    """Run the performance test for the defined scenarios."""
    for benchmark in BENCHMARKS:
        benchmark.run()


if __name__ == "__main__":
    main()
