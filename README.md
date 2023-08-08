# BayBE — A Bayesian Back End for Design of Experiments

This software package provides a general-purpose toolbox for **Design of Experiments
(DOE)**.

It provides the necessary functionality to:

- manage experimental design spaces that can be defined through various types of
  experimental parameters.
- execute different kinds of DOE strategies, levering both classical and
  machine-learning-based models.
- handle measurements data and feed it back into the experimental design.
- compare different DOE strategies through backtesting with synthetic and real data.

## :exclamation::construction: NOTE :construction::exclamation:

This repository is under **heavy active development**.
Please note that the provided functionality and user interfaces are not stable and may
change in newer releases.
Therefore, if you would like to use the code in its current early stage, we recommend
pinning the version during installation to prevent possible changes in the backend.
In case of questions or comments, feel free to reach out to the **BayBE Dev Team** (see
[pyproject.toml](./pyproject.toml) for contact details).

## Installation

There are several ways to install BayBE.
Essentially, these can be divided into 1) installation from Azure Artifacts via pip
or 2) direct installation from this repository using git.

### Installation from Azure Artifacts

If a specific **released** BayBE version is to be installed, this can be achieved via
[Azure Artifacts](https://***REMOVED***/_artifacts/feed/artifacts).

The most convenient way for this is to generate a personal access token (PAT) on the
[user settings page](https://***REMOVED***/_usersSettings/tokens).
Once the token is created, BayBE can be installed via

```bash
pip install --extra-index-url https://artifacts:<token>@pkgs.***REMOVED***/_packaging/artifacts/pypi/simple/ baybe
```

where `<token>` needs to be replaced with your PAT.
Note that, instead of passing the `--extra-index-url` option directly as an argument
to the `install` command, you can alternatively create a `pip.conf` file containing
the same information.

A second way to authenticate instead of using a PAT is via `artifacts-keyring`.
To do so, ensure that all [prerequisites](https://pypi.org/project/artifacts-keyring/)
are fulfilled and run:

```bash
pip install keyring artifacts-keyring
pip install --extra-index-url https://pkgs.***REMOVED***/_packaging/artifacts/pypi/simple/ baybe
```

### Installation from Repository

If you need finer control and would like to install a specific commit that has not been
released under a certain version tag, you can do so by installing BayBE directly from
the repository.
First, clone the repository, navigate to the repository root folder, check out the
desired commit, and run:

```bash
pip install .
```

There are additional dependencies that can be installed corresponding to linters, 
plotters etc. (`dev`) and scikit-learn-extra (`extra`) which is not yet available 
on osx-arm64. A developer would typically also install the package in editable mode 
('-e').

```bash
pip install -e '.[dev,extra]'
```

## Getting Started

BayBE is a DOE software built to streamline your experimental process.
It can process measurement data from previous experiments and, based on these, provide
optimal experimental designs to further improve your target quantities.

In order to make use of BayBE's optimization capabilities, you need to translate your
real-world optimization problem into mathematical language.
To do so, you should ask yourself the following questions:

* What should be optimized?
* What are the degrees of freedom?
* (Optional) What optimization strategy should be used?

Conveniently, the answer to each of these questions can be directly expressed in the
form of objects in BayBE's ecosystem that can be easily mixed and matched:

| Part of the Problem Specification                     | Defining BayBE Objects    |
|:------------------------------------------------------|:--------------------------|
| What should be optimized?                             | `Objective`, `Target`     |
| What are the degrees of freedom?                      | `Parameter`, `Constraint` | 
| (Optional) What optimization strategy should be used? | `Strategy`, `Recommender` |

The objects in the first two table rows can be regarded as embodiments of the
**mathematical DOE specifications** in digital form, as they fully define the underlying
optimization problem.
By contrast, the objects in the last row rather provide **algorithmic details**
on how the DOE problem should be solved.
In that sense, the former carry information that **must be** provided by the user,
whereas the latter are **optional** settings that can also be set automatically
by BayBE.

The following example provides a step-by-step guide to what this translation process
should look like, and how we can subsequently use BayBE to generate optimal sets of
experimental conditions.

### Defining the Optimization Objective

We start by defining an optimization objective.
While BayBE ships with the necessary functionality to optimize multiple targets
simultaneously, as an introductory example, we consider a simple scenario where our
goal is to **maximize** a single numerical target that represents the yield of a
chemical reaction.

In BayBE's language, the reaction yield can be represented as a `NumericalTarget`
object:

```python
from baybe.targets import NumericalTarget

target = NumericalTarget(
    name="Yield",
    mode="MAX",
)
```

We wrap the target object in an optimization `Objective`, to inform BayBE
that this is the only target we would like to consider:

```python
from baybe.targets import Objective

objective = Objective(mode="SINGLE", targets=[target])
```

In cases where we need to consider multiple (potentially competing) targets, the
role of the `Objective` is to define how these targets should be balanced.
For more details, see [baybe/targets.py](./baybe/targets.py).

### Defining the Search Space

Next, we inform BayBE about the available "control knobs", that is, the underlying
system parameters we can tune to optimize our targets.
This also involves specifying their ranges and other parameter-specific details.

For our reaction example, we assume that we can control the following three quantities:

```python
from baybe.parameters import CategoricalParameter, NumericalDiscreteParameter, SubstanceParameter

parameters = [
    CategoricalParameter(
        name="Granularity",
        values=["coarse", "medium", "fine"],
        encoding="OHE",
    ),
    NumericalDiscreteParameter(
        name="Pressure[bar]",
        values=[1, 5, 10],
        tolerance=0.2,
    ),
    SubstanceParameter(
        name="Solvent",
        data={"Solvent A": "COC", "Solvent B": "CCC", "Solvent C": "O",
              "Solvent D": "CS(=O)C"},
        encoding="MORDRED",
    ),
]
```

Note that each parameter is of a different **type** and thus requires its own
type-specific settings. In particular case above, for instance:

* `encoding=OHE` activates one-hot-encoding for the categorical parameter "Granularity".
* `tolerance=0.2` allows experimental inaccuracies up to 0.2 when reading values for
  "Pressure[bar]".
* `encoding=MORDRED`triggers computation of MORDRED cheminformatics descriptors for
  the substance parameter "Solvent".

For more parameter types and their details, see
[baybe/parameters.py](./baybe/parameters.py).

Additionally, we can define a set of constraints to further specify allowed ranges and
relationships between our parameters.
Details can be found in [baybe/constraints.py](./baybe/constraints.py).
In this example, we assume no further constraints and explicitly indicate this with an
empty variable, for the sake of demonstration:

```python
constraints = None
```

With the parameter and constraint definitions at hand, we can now create our
`SearchSpace`:

```python
from baybe.searchspace import SearchSpace

searchspace = SearchSpace.from_product(parameters, constraints)
```

### Optional: Defining the Optimization Strategy

As an optional step, we can specify details on how the optimization should be
conducted.
If omitted, BayBE will choose a default setting.

For our chemistry example, we combine two selection strategies:

1. In cases where no measurements have been made prior to the interaction with BayBE,
   a random experiment selection strategy is used to produce initial recommendations.
2. As soon as the first measurements are available, we switch to a Bayesian approach
   where points are selected greedily from a probabilistic prediction model.

For more details on the different strategies, their underlying algorithmic
details, and their configuration settings, see
[baybe/strategies](./baybe/strategies).

```python
from baybe.strategies import Strategy, SequentialGreedyRecommender, RandomRecommender

strategy = Strategy(
    initial_recommender=RandomRecommender(),
    recommender=SequentialGreedyRecommender(),
)
```

### The Optimization Loop

Having provided the answers to [all questions above](#getting-started), we can now
construct a BayBE object that brings all
pieces of the puzzle together:

```python
from baybe import BayBE

baybe = BayBE(searchspace, objective, strategy)
```

With this object at hand, we can start our experimentation cycle.
In particular:

* We can ask BayBE to `recommend` new experiments.
* We can `add_measurements` for certain experimental settings to BayBE's database.

Note that these two steps can be performed in any order.
In particular, available measurement data can be submitted at any time.
Also, we can start the interaction with either command and repeat the same type of
command immediately after its previous call, e.g., if the required number of
recommendations has changed.

The following illustrates one such possible sequence of interactions.
Let us first ask for an initial set of recommendations:

```python
df = baybe.recommend(batch_quantity=5)
```

For a particular random seed, the result could look as follows:

| Granularity   | Pressure[bar]   | Solvent   |
|---------------|-----------------|-----------|
| medium        | 1               | Solvent B |
| medium        | 5               | Solvent D |
| fine          | 5               | Solvent C |
| fine          | 5               | Solvent A |
| medium        | 10              | Solvent B |

After having conducted the corresponding experiments, we can add our measured
yields to the table and feed it back to BayBE:

```python
df["Yield"] = [79, 54, 59, 95, 84]
baybe.add_measurements(df)
```

With the newly arrived data, BayBE will update its internal state and can produce a
refined design for the next iteration.

## Telemetry

By default, BayBE collects anonymous usage statistics.
Note that this does **not** involve logging of recorded measurements, targets or any
project information that would allow the reconstruction of details.

Monitored quantities are:

- `batch_quantity` used when querying recommendations
- number of parameters in the search space
- number of constraints in the search space
- how often `recommend` was called
- how often `add_measurements` was called
- how often a search space is newly created
- how often initial measurements are added before recommendations were calculated
  ("naked initial measurements")
- the fraction of measurements added that correspond to previous recommendations

These metrics are vital to demonstrating the impact of the project and
– should you find BayBE useful – we kindly ask you to leave telemetry activated.
If you wish to disable logging, you can set the following environment variable:

```bash
export BAYBE_TELEMETRY_ENABLED=false
```

or in python:

```python
import os

os.environ["BAYBE_TELEMETRY_ENABLED"] = "false"
```

before calling any BayBE functionality.
Telemetry can be re-enabled by simply removing the variable:

```bash
unset BAYBE_TELEMETRY_ENABLED
```

or in python:

```python
os.environ.pop["BAYBE_TELEMETRY_ENABLED"]
```

Note, however, that (un-)setting the variable in the shell will not affect the running
python session.

## License

Copyright 2022-2023 Merck KGaA, Darmstadt, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
