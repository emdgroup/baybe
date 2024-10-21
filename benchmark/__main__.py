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

    The following function checks if persisting data to S3 is enabled.
    If not, the function will use the local file system to store the data.
    This is meant to be used for development and local testing only.

    Parameters:
        time_stamp_test_execution: The timestamp of the test.

    Returns:
        ResultPersistenceInterface: An instance of ResultPersistenceInterface.

    """
    if os.environ.get("BAYBE_PERFORMANCE_PERSISTANCE_PATH"):
        return S3ExperimentResultPersistence(time_stamp_test_execution)
    return LocalExperimentResultPersistence(time_stamp_test_execution)


def main():
    """Run the performance test for the given scenario.

    This function runs the performance test cases defined in the domain module
    in parallel. The function uses the number of cores available on the machine
    to create a reasonable number of workers for the execution.
    The results of the benchmarks are persisted to the file system or S3.
    """
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
