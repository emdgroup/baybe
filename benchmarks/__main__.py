"""Executes the benchmarking module."""

from benchmarks.domains import BENCHMARKS


def main():
    """Run all benchmarks."""
    for benchmark in BENCHMARKS:
        benchmark()


if __name__ == "__main__":
    main()
