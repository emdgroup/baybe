"""Executes the benchmarking module."""
# Run this via 'python -m benchmarks' from the root directory.

from benchmarks.domains import BENCHMARKS
from benchmarks.persistence import make_object_writer, make_path_constructor


def main():
    """Run all benchmarks."""
    persistence_handler = make_object_writer()

    for benchmark in BENCHMARKS:
        result = benchmark()
        path_constructor = make_path_constructor(benchmark, result)
        persist_dict = benchmark.to_dict() | result.to_dict()
        persistence_handler.write_json(persist_dict, path_constructor)


if __name__ == "__main__":
    main()
