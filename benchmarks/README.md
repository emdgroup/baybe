This module contains benchmarks meant to test the performance of BayBE for
pre-defined tasks. The benchmarks can be executed as a whole by executing
the following command:

```bash
python -m benchmarks
```

An alternative would be to call only a subset of the benchmarks by providing
one or multiple names from the benchmarks in the list
 `BENCHMARKS` from the `__init__.py` file in the
`domains` folder. More about the `BENCHMARKS` list can be found in the section
[Add your benchmark to the benchmarking module]
(#add-your-benchmark-to-the-benchmarking-module).
A subset of benchmarks where the benchmark `synthetic_2C1D_1C` would be the first in
the list can be called with:

```bash
python -m benchmarks --benchmark-list synthetic_2C1D_1C
```

Please find instruction on how to add the benchmarks to the CI/CD pipeline in the
section [Add benchmark to CI/CD pipeline](#add-benchmark-to-ci/cd-pipeline).

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

## Add benchmark to CI/CD pipeline

The benchmarks will not automatically be executed in the CI/CD pipeline.
You have to provide them as clickable inputs in the GitHub Actions workflow.
To do this, a new Boolean checkbox needs to be added to the workflow file
`manual_benchmark.yml` in the`.github/workflows` folder. The checkbox name must exactly
match the benchmark name. The benchmark `synthetic_2C1D_1C.py` with the callable
`synthetic_2C1D_1C` (which is named after the callable NOT the file) would look
like this:

```yaml
      synthetic_2C1D_1C:
        description: "Run synthetic_2C1D_1C benchmark"
        required: false
        default: false
        type: boolean
```

which can just be copied below the existing checkboxes. If we would add a benchmark
called `foo` with the callable `bar` the checkbox would look like this:

```yaml
name: Run Benchmark

on:
  workflow_dispatch:
    inputs:
      group_selection:
        description: "Select the group of benchmarks to run"
        required: true
        default: "All"
        type: choice
        options:
          - "Manually Selected"
          - "All"
          - "Default"
      synthetic_2C1D_1C:
        description: "Synthetic_2C1D_1C benchmark"
        required: false
        default: false
        type: boolean
      bar:
        description: "Foo benchmark"
        required: false
        default: false
        type: boolean
```

### Add benchmark group

There are also groups that can be defined in the workflow file. The benchmarks in the
groups are jointly executed when the group is selected. Groups are defined as env variables
in the workflow file. If you want to add `bar` to the `DEFAULT_BENCHMARKS` group, you
can just write it behind the existing entities separated by a comma:

```yaml
DEFAULT_BENCHMARKS: '["synthetic_2C1D_1C", "bar"]'
```

To add a new group `FOO_BAR` with `foo`, you can define it below the existing groups:

```yaml
env:
    DEFAULT_BENCHMARKS: '["synthetic_2C1D_1C"]'
    FOO_BAR: '["foo"]'
```

You also have to add it to the dropdown menu in the workflow file:

```yaml
      group_selection:
        description: "Select the group of benchmarks to run"
        required: true
        default: "All"
        type: choice
        options:
          - "Manually Selected"
          - "All"
          - "Default"
          - "Foo bar"             #<-- Add this line
```

so that the selected group can be checked in the step `build_matrix_from_group`, where
you have to add:

```yaml
          if [ "$run_all_benchmarks" = "Default" ]; then
            benchmarks_to_execute='{"benchmark_list": ${{ env.DEFAULT_BENCHMARKS }} }'
          elif [ "$run_all_benchmarks" = "Foo bar" ]; then                   #<-- Add this line
            benchmarks_to_execute='{"benchmark_list": ${{ env.FOO_BAR }} }'  #<-- Add this line
          fi
``` 
