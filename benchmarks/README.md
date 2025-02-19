This module contains benchmarks meant to test the performance of BayBE for
pre-defined tasks. The benchmarks can be executed as a whole by executing
the following command:

```bash
python -m benchmarks
```

# `Benchmark`

The `Benchmark` object is the combination of all benchmark related data.
At the heart is the callable `function`, used to perform and hide the
benchmarked code. The `name` serves as the unique identifier of the benchmark. Note that
this identifier is also used for storing a `Result`. Therefore, any change will be
considered a new benchmark. The `function`s `__doc__` is used to
automatically set the `description`. A full code example can be found in the
`domains/synthetic_2C1D_1C.py` file.

# `BenchmarkSettings`

The `BenchmarkSettings` object is used to parameterize the benchmark `function`.
It is an abstract base class that can be extended by the user to provide
additional information. The only required attribute is
`random_seed`, which is used to seed the entire call of the benchmark `function`.
Currently, the following settings are available:

## `ConvergenceExperimentSettings`

The `ConvergenceExperimentSettings` object is used to parameterize the
convergence experiment benchmarks and holds information used for BayBE scenario
executions. Please refer to the BayBE documentation for more information
about the [simulations subpackage](baybe.simulation).

# `Result`

The `Result` object encapsulates all execution-relevant information of the `Benchmark`
and represents the `Result` of the benchmark `function`, along with state information
at the time of execution.

## `ResultMetadata`

The `ResultMetadata` is the wrapper to hold the described information about the
`Benchmark` at runtime. A combination of the benchmark identifier and the metadata
is meant to describe the conducted `Result` uniquely under the assumption that equal
benchmarked code states are also equally representative due to the fixed random seed.

# Add your benchmark to the benchmarking module

In the last step, your benchmark object has to be added to the
`benchmarks module`. This is done by adding the object to the `BENCHMARKS`
list in the `__init__.py` file in the `domains` folder. The `BENCHMARKS` contains all
objects that should be called when running the `benchmarks module`.

# Persisting Results

`Result`s are stored automatically. Since multiple storage types are provided with
different requirements and compatibilities, the `PathConstructor` class is used to
construct the identifier for the file. For example `S3ObjectStorage` is used to
store the `Result`s in an S3 bucket which separates the key by `/` but does not create
real folders while the usual local persistence creates a file with a `_` so that folder
creation is not necessary. The class handling the storage of the resulting object get
this `PathConstructor` and use it in the way it needs the identifier to be.
The following types of storage are available:

## `LocalFileObjectStorage`

Stores a file on the local file system and will automatically be chosen when calling
the `benchmarks module` if it does not run in the CI/CD pipeline. A prefix folder path can be
provided when creating the object. The file will be stored in the current working
directory if no prefix is provided. The file will be stored in the following format
with the prefix:
`<PREFIX_PATH>/<benchmark_name>_<branch>_<latest_baybe_tag>_<execution-date>_<commit_hash>_result.json`.

## `S3ObjectStorage`

Stores a file in an S3 bucket and will automatically be chosen when calling the
`benchmarks module` if it runs in the CI/CD pipeline. For locating the S3-Bucket to
persist, the environment variable `BAYBE_BENCHMARKING_PERSISTENCE_PATH` must be set
with its name. For running the `benchmarks module` in the CI/CD pipeline,
there must be also the possibility to assume a AWS role from a job call.
This is done by providing the roles ARN in the secret `AWS_ROLE_TO_ASSUME`.
For creating temporary credentials, a GitHub App will be used.
To generated a token, the id of the GitHub App and its secret key must be provided in
the secrets `APP_ID` and `APP_PRIVATE_KEY`. The file will be stored in the following
format: `<benchmark_name>/<branch>/<latest_baybe_tag>/<execution-date>/<commit_hash>/result.json`.
