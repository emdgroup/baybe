"""Run the benchmarks for the given scenario."""

import concurrent.futures
import os
from datetime import datetime

from benchmark.domain import SINGE_BENCHMARKS_TO_RUN
from benchmark.src import (
    LocalExperimentResultPersistence,
    ResultPersistenceInterface,
    S3ExperimentResultPersistence,
    SingleResult,
)


def result_data_handler(
    time_stamp_test_execution: datetime,
) -> ResultPersistenceInterface:
    """Create a result data handler.

    This fixture is used to store the results of the performance tests in a persistent
    way with a function scope to ensure that all test cases can run independently
    since basic boto3 is not thread safe but creating a boto3 client session is.
    For local testing, the results are stored in a local directory.

    Parameters:
        time_stamp_test_execution: The timestamp of the test.

    Returns:
        ResultPersistenceInterface: An instance of ResultPersistenceInterface.

    """
    if os.environ.get("BAYBE_PERFORMANCE_PERSISTANCE_PATH"):
        return S3ExperimentResultPersistence(time_stamp_test_execution)
    return LocalExperimentResultPersistence(time_stamp_test_execution)


def main():
    """Run the performance test for the given scenario."""
    num_cores = os.cpu_count()
    persistance_handler = result_data_handler(datetime.now())
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [
            executor.submit(func.execute_benchmark) for func in SINGE_BENCHMARKS_TO_RUN
        ]
        for future in concurrent.futures.as_completed(futures):
            result_benchmarking: SingleResult = future.result()
            persistance_handler.persist_new_result(result_benchmarking)


if __name__ == "__main__":
    main()
