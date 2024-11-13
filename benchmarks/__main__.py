"""Executes the benchmarking module."""
# Run this via 'python -m benchmarks' from the root directory.

from benchmarks.domains import BENCHMARKS
from benchmarks.persistence import create_persistence_object, persister_factory


def main():
    """Run all benchmarks."""
    persistence_handler = persister_factory()

    for benchmark in BENCHMARKS:
        result = benchmark()
        persistence_object = create_persistence_object(benchmark, result)
        persistence_handler.write_object(persistence_object)


if __name__ == "__main__":
    main()
