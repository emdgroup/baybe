"""Executes the benchmarking module."""
# Run this via 'python -m benchmarks' from the root directory.

from benchmarks.domains import BENCHMARKS


def main():
    """Run all benchmarks."""
    for benchmark in BENCHMARKS:
        benchmark()


if __name__ == "__main__":
    main()
