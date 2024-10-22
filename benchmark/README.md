# Description

This folder contains performance tests which reflect long running experimental scenarios. They are not meant to be run manually since their execution time is typically long. However, if you want to run them, you can do so by executing the following tox command:

```bash
python -m benchmark
```

These tests should reflect the performance of the codebase and should be used to identify performance regressions which may only visible in long running scenarios. We refer to performance in a quality oriented way, meaning that we are especially interested in the convergence behavior of the provided algorithms and strategies. The performance tests are not meant to be used for classical compute benchmarking purposes, but rather to ensure that the codebase is able to solve the given optimization problems in a consistent and reliable way. To compare the performance of different versions of the codebase properly, we store the results of the performance tests either as csv files if run locally or in a database if run on a CI system.

## Adding a new test case

Each scenario type of BayBE is represented by a separate test case. These are collected in the `domain` folder where a file for each test scenario where the benchmark code is defined. The benchmark object must be added to the `__init__.py` file in the `domain` folder to be recognized by the test runner. If lookup tables are necessary, they can be added to the `domain/lookup` folder and load with a provided prefix from the `utils.py` file (See `domain/direct_arylation.py`). The benchmark class expects a callable which returns a `result` dataframe from a scenario execution by BayBE and a dictionary with strings as keys and values to reflect metadata. Furthermore a unique uuid is necessary and can be generated online under (<https://www.uuidgenerator.net/>). However, this is only necessary if the results are stored in a database to compare them afterward. The title can be chosen at will. The benchmark could look like this:

```python
"""Synthetic dataset. Custom parabolic test with irrelevant parameters."""

from uuid import UUID

from pandas import DataFrame, read_csv

from benchmark.src import SingleExecutionBenchmark
from benchmark.domain.utils import PATH_PREFIX

def example_benchmark_function() -> tuple[DataFrame, dict[str, str]]:
    """Synthetic dataset. Custom parabolic test with irrelevant parameters."""
    example_benchmark_lookup = read_csv(
        PATH_PREFIX.joinpath("example.csv").resolve()
    )
    [...]
    some BayBE code
    [...]

    batch_size = 5
    n_doe_iterations = 30
    n_mc_iterations = 50

    metadata = {
        "DOE_iterations": str(n_doe_iterations),
        "batch_size": str(batch_size),
        "n_mc_iterations": str(n_mc_iterations),
    }

    scenarios = {
        "Default Two Phase Meta Recommender": campaign,
        "Random Baseline": campaign_rand,
    }
    return simulate_scenarios(
        scenarios,
        lookup_synthetic_1,
        batch_size=batch_size,
        n_doe_iterations=n_doe_iterations,
        n_mc_iterations=n_mc_iterations,
    ), metadata


example_benchmark = SingleExecutionBenchmark(
    title="Example benchmark.",
    identifier=UUID("1858e128-29ba-4be5-8100-2698fd54fac2"),
    benchmark_function=example_benchmark_function,
)
```

Where `example_benchmark` is added to the list `SINGE_BENCHMARKS_TO_RUN` in the `__init__.py` file in the `domain` folder. `__main__.py` will use this list and run all entries in parallel if called.

## Persisting test results

For persisting the test result in the CI/CD, currently a S3-Bucket name is used. This name must be set in the environment variable `BAYBE_PERFORMANCE_PERSISTANCE_PATH` in the CI/CD system. The test results are stored under the key `ORG_STARTING_WORKFLOW/UUID/BRANCH_NAME/BAYBE_VERSION/DATE_TIME_ISO/COMMIT_HASH/WORKFLOW-ID` as a csv file. The created S3 entry contains information like the batch size and the number of DOE iterations, but also the execution time and the title of the test case. If the environment variable is not set, the test results are stored locally in the `results` folder. The test results are stored as csv files with the name `UUID-BAYBE_VERSION-DATE_TIME_ISO.csv` without additional metadata.

### How metrics work

The usage of metrics is optional and currently only a preparation for further development. They can be created with an optional threshold. If and only if the threshold us set and the metric is violated, the benchmark will throw an exception. Due to the different scenarios in one dataframe storing results, a metric returns either a dict with all results or only those which are specified in a list given. The same applies to the treshholds which are defined by the name of the scenario (`"Default Two Phase Meta Recommender"` and `"Random Baseline"` in the example above). The metric could look like this:

```python
from benchmark.src.metric import NormalizedAreaUnderTheCurve
[...]

min_may_yield = (0.0, 15.0)

auc = NormalizedAreaUnderTheCurve(
    lookup=min_may_yield,
    objective_name="yield",
    threshold={"Default Two Phase Meta Recommender": 0.5},
)

example_benchmark = SingleExecutionBenchmark(
    title="Example benchmark.",
    identifier=UUID("1858e128-29ba-4be5-8100-2698fd54fac2"),
    benchmark_function=example_benchmark_function,
    metrics=[auc],
    objective_scenarios=["Default Two Phase Meta Recommender"]
)

result = example_benchmark.execute_benchmark()
auc.evaluate(result, ["Random Baseline"])                       # Evaluation can also be done after the benchmark execution
```
