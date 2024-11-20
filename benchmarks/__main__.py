"""Executes the benchmarking module."""
# Run this via 'python -m benchmarks' from the root directory.

from benchmarks.domains import BENCHMARKS
from benchmarks.persistence import persistence_object_factory, persister_factory


def main():
    """Run all benchmarks."""
    persistence_handler = persister_factory()

    for benchmark in BENCHMARKS:
        result = benchmark()
        persistence_object = persistence_object_factory(benchmark, result)
        persist_dict = benchmark.to_dict() | result.to_dict()
        persistence_handler.write_object(persistence_object, persist_dict)


if __name__ == "__main__":
    main()
