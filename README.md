<div align="center">
  <br/>

[![CI](https://img.shields.io/github/actions/workflow/status/emdgroup/baybe/ci.yml?branch=main&style=flat-square&label=CI&labelColor=0f69af&color=ffdcb9)](https://github.com/emdgroup/baybe/actions/workflows/ci.yml?query=branch%3Amain)
[![Regular](https://img.shields.io/github/actions/workflow/status/emdgroup/baybe/regular.yml?branch=main&style=flat-square&label=Regular%20Check&labelColor=0f69af&color=ffdcb9)](https://github.com/emdgroup/baybe/actions/workflows/regular.yml?query=branch%3Amain)
[![Docs](https://img.shields.io/github/actions/workflow/status/emdgroup/baybe/docs.yml?branch=main&style=flat-square&label=Docs&labelColor=0f69af&color=ffdcb9)](https://github.com/emdgroup/baybe/actions/workflows/docs.yml?query=branch%3Amain)

[![Supports Python](https://img.shields.io/pypi/pyversions/baybe?style=flat-square&label=Supports%20Python&labelColor=96d7d2&color=ffdcb9)](https://pypi.org/project/baybe/)
[![PyPI version](https://img.shields.io/pypi/v/baybe.svg?style=flat-square&label=PyPI%20Version&labelColor=96d7d2&color=ffdcb9)](https://pypi.org/project/baybe/)
[![Downloads](https://img.shields.io/pypi/dm/baybe?style=flat-square&label=Downloads&labelColor=96d7d2&color=ffdcb9)](https://pypistats.org/packages/baybe)
[![Issues](https://img.shields.io/github/issues/emdgroup/baybe?style=flat-square&label=Issues&labelColor=96d7d2&color=ffdcb9)](https://github.com/emdgroup/baybe/issues/)
[![PRs](https://img.shields.io/github/issues-pr/emdgroup/baybe?style=flat-square&label=PRs&labelColor=96d7d2&color=ffdcb9)](https://github.com/emdgroup/baybe/pulls/)
[![License](https://shields.io/badge/License-Apache%202.0-green.svg?style=flat-square&labelColor=96d7d2&color=ffdcb9)](http://www.apache.org/licenses/LICENSE-2.0)

[![Logo](https://raw.githubusercontent.com/emdgroup/baybe/main/docs/_static/banner2.svg)](https://github.com/emdgroup/baybe/)

&nbsp;
<a href="https://emdgroup.github.io/baybe/">Homepage<a/>
&nbsp;•&nbsp;
<a href="https://emdgroup.github.io/baybe/stable/userguide/userguide.html">User Guide<a/>
&nbsp;•&nbsp;
<a href="https://emdgroup.github.io/baybe/stable/_autosummary/baybe.html">Documentation<a/>
&nbsp;•&nbsp;
<a href="https://emdgroup.github.io/baybe/stable/misc/contributing_link.html">Contribute<a/>
&nbsp;
</div>

# BayBE — A Bayesian Back End for Design of Experiments

The **Bay**esian **B**ack **E**nd (**BayBE**) 
helps to find a **good parameter setting** 
within a complex parameter search space. 

<div align="center">

![complex search space](docs/_static/complex_search_space_automatic.svg)

</div>

Example use-cases:
- 🧪 Find chemical reaction conditions or process parameters
- 🥣 Create materials, chemical mixtures, or formulations with desired properties
- ✈️ Optimize the 3D shape of a physical object
- 🖥️ Select model hyperparameters
- 🫖 Find tasty espresso machine settings

This is achieved via **Bayesian Design of Experiments**, 
which is an efficient way for navigating parameter search spaces. 
It balances
exploitation of parameter space regions known to lead to good outcomes 
and exploration of unknown regions. 

BayBE provides a **general-purpose toolbox** for Bayesian Design of Experiments, 
focusing on making this procedure easily-accessible for real-world experiments.

## 🔋 Batteries Included
BayBE offers a range of ✨**built&#8209;in&nbsp;features**✨ crucial for real-world use cases.
The following provides a non-comprehensive overview:

- 📚 Leverage **domain knowledge**.  
  - 🎨 Encode categorical data to capture relationships between categories. BayBE also provides built-in chemical encodings.
  - 🛠️ Option to build-in mechanistic process understanding via custom surrogate models.
- 🏛️ Leverage **historic data** to accelerate optimization via transfer learning.
- 🌀 **Flexible** definition of target outcomes, parameter search spaces, and optimization strategies:
  - 🎯 Option to use numerical targets (e.g., experimental outcome values) or binary targets (e.g., good/bad classification of experimental results). Targets can be minimized, maximized, or matched to a specific value.
  - 👥👥 Multiple targets can be optimized at once (e.g., via Pareto optimization).
  - 🎭 Both continuous and discrete parameters can be used within a single search space.
  - 🔢 The maximal number of mixture components can be defined via cardinality constraints.
  - ⚖️ Different optimization strategies can be selected to balance exploration and exploitation of the search space, including bandit models and active learning.
- 🌎 Run campaigns **asynchronously** with pending experiments and partial measurements via distributed workflows.
- 🔍 **Insights**: Easily analyze feature importance and model behavior.
- 📈 Utilities for **benchmarking**, such as backtesting and simulations.
- 📝 **High-quality code base** with comprehensive tests and typing.
- 🔄 Code is designed with **database storage and API** wrappers in mind via serialization.


## ⚡ Quick Start

To perform Bayesian Design of Experiments with BayBE, 
the users must first specify the **parameter search space** and **objective** to be optimized. 
Based on this information and any **available data** about outcomes of specific parameter settings, 
BayBE will **recommend the next set of parameter combinations** to be **measured**. 
To inform the next recommendation cycle, the newly generated measurements can be added to BayBE.

<div align="center">

![quick start](docs/_static/quick_start_automatic.svg)

</div>

From the user-perspective, the most important part is the "design" step.
If you are new to BayBE, we suggest consulting our 
[design checklist](https://emdgroup.github.io/baybe/stable/faq.html#checklist-for-designing-baybe-experiments) 
to help you with the design setup.

Below we show a simple optimization procedure, starting with the design step and subsequently
performing the recommendation loop. 
The provided example aims to maximize the yield of a chemical reaction by adjusting reaction parameters.

First, install BayBE into your Python environment: 
```bash 
pip install baybe 
``` 
For more information on this step, see our
[detailed installation instructions](#installation).

### Defining the Optimization Objective

In BayBE's language, the reaction yield can be represented as a `NumericalTarget`,
which we wrap into a `SingleTargetObjective`:

```python
from baybe.targets import NumericalTarget
from baybe.objectives import SingleTargetObjective

target = NumericalTarget(name="Yield")
objective = SingleTargetObjective(target=target)
```
In cases where we are confronted with multiple (potentially conflicting) targets 
(e.g., yield vs cost),
the `ParetoObjective` or `DesirabilityObjective` can be used instead.
These allow to define additional settings, such as how the targets should be balanced.
For more details, see the
[objectives section](https://emdgroup.github.io/baybe/stable/userguide/objectives.html)
of the user guide.

### Defining the Search Space

Next, we inform BayBE about the available "control knobs", that is, the underlying
reaction `Parameters` we can tune (e.g., granularity,
pressure, and solvent) to optimize the yield. We also need to specify
which values individual parameters can take.

```python
from baybe.parameters import (
    CategoricalParameter,
    NumericalDiscreteParameter,
    SubstanceParameter,
)

parameters = [
    CategoricalParameter(
        name="Granularity",
        values=["coarse", "medium", "fine"],
        encoding="OHE",  # one-hot encoding of categories
    ),
    NumericalDiscreteParameter(
        name="Pressure[bar]",
        values=[1, 5, 10],
        tolerance=0.2,  # allows experimental inaccuracies up to 0.2 when reading values
    ),
    SubstanceParameter(
        name="Solvent",
        data={
            "Solvent A": "COC",
            "Solvent B": "CCC",  # label-SMILES pairs
            "Solvent C": "O",
            "Solvent D": "CS(=O)C",
        },
        encoding="MORDRED",  # chemical encoding via scikit-fingerprints
    ),
]
```

For more parameter types and their details, see the
[parameters section](https://emdgroup.github.io/baybe/stable/userguide/parameters.html)
of the user guide.

Additionally, we can define a set of constraints to further specify allowed ranges and
relationships between our parameters. Details can be found in the
[constraints section](https://emdgroup.github.io/baybe/stable/userguide/constraints.html) of the user guide.
In this example, we assume no further constraints.

With the parameter definitions at hand, we can now create our
`SearchSpace` based on the Cartesian product of all possible parameter values:

```python
from baybe.searchspace import SearchSpace

searchspace = SearchSpace.from_product(parameters)
```

See the [search spaces section](https://emdgroup.github.io/baybe/stable/userguide/searchspace.html)
of our user guide for more information on the structure of search spaces
and alternative ways of construction. 

### Optional: Defining the Optimization Strategy

As an optional step, we can specify details on how the optimization of the experiment should be
performed. If omitted, BayBE will choose a default setting.

For our example, we combine two recommenders via a so-called meta recommender named
`TwoPhaseMetaRecommender`:

1. In cases where no measurements have been made prior to the interaction with BayBE,
   the parameters will be recommended with the `initial_recommender`.
2. As soon as the first measurements are available, we switch to the `recommender`.

```python
from baybe.recommenders import (
    BotorchRecommender,
    FPSRecommender,
    TwoPhaseMetaRecommender,
)

recommender = TwoPhaseMetaRecommender(
    initial_recommender=FPSRecommender(),  # farthest point sampling
    recommender=BotorchRecommender(),  # Bayesian model-based optimization
)
```

For more details on the different recommenders, their underlying algorithmic
details, and their configuration settings, see the
[recommenders section](https://emdgroup.github.io/baybe/stable/userguide/recommenders.html)
of the user guide.

### The Optimization Loop

We can now construct a `Campaign` that performs the Bayesian optimization of the experiment:

```python
from baybe import Campaign

campaign = Campaign(searchspace, objective, recommender)
```

With this object at hand, we can start our experimentation cycle.
In particular:

* The campaign can `recommend` new experiments.
* We can `add_measurements` of target values for the measured parameter settings 
  to the campaign's database.

Note that these two steps can be performed in any order.
In particular, available measurements can be submitted at any time and also several 
times before querying the next recommendations.

```python
df = campaign.recommend(batch_size=3) # Recommend three parameter settings
print(df)
```

The below table shows the three parameter setting for which BayBE recommended to 
measure the reaction yield.

Note that the specific recommendations will depend on both the data
already fed to the campaign and the random number generator seed that is used.

```none
   Granularity  Pressure[bar]    Solvent
15      medium            1.0  Solvent D
10      coarse           10.0  Solvent C
29        fine            5.0  Solvent B
```

After having conducted the recommended experiments, we can add the newly measured
target information to the campaign:

```python
df["Yield"] = [79.8, 54.1, 59.4] # Measured yields for the three recommended parameter settings
campaign.add_measurements(df)
```

With the newly provided data, BayBE can produce a refined recommendation for the next iteration.
This loop typically continues until a desired target value is achieved in the experiment.

### Inspect the progress of the experiment optimization

The below plot shows progression of a campaign that optimized direct arylation reaction
by tuning the solvent, base and ligand 
(from [Shields, B.J. et al.](https://doi.org/10.1038/s41586-021-03213-y)).
Each line shows the best target value that was measured in each experimental iteration.
Different lines show outcomes of `Campaigns` with different designs.

![Substance Encoding Example](./examples/Backtesting/full_lookup_light.svg)

In particular, the five `Campaigns` differ in how the chemical `Parameters` were encoded.
We can see that optimization is more efficient when 
using chemical encodings (e.g., *MORDRED*) rather than encoding categories with *one-hot* encoding or *random* features.

(installation)=
## 💻 Installation
### From Package Index
The easiest way to install BayBE is via PyPI:

```bash
pip install baybe
```

A certain released version of the package can be installed by specifying the
corresponding version tag in the form `baybe==x.y.z`.

### From GitHub
If you need finer control and would like to install a specific commit that has not been
released under a certain version tag, you can do so by installing BayBE directly from
GitHub via git and specifying the corresponding
[git ref](https://pip.pypa.io/en/stable/topics/vcs-support/#git).

For instance, to install the latest commit of the main branch, run:

```bash
pip install git+https://github.com/emdgroup/baybe.git@main
```


### From Local Clone

Alternatively, you can install the package from your own local copy.
First, clone the repository, navigate to the repository root folder, check out the
desired commit, and run:

```bash
pip install .
```

A developer would typically also install the package in editable mode ('-e'),
which ensures that changes to the code do not require a reinstallation.

```bash
pip install -e .
```

If you need to add additional dependencies, make sure to use the correct syntax
including `''`:

```bash
pip install -e '.[dev]'
```

### Optional Dependencies
There are several dependency groups that can be selected during pip installation, like
```bash
pip install 'baybe[test,lint]' # will install baybe with additional dependency groups `test` and `lint`
```
To get the most out of `baybe`, we recommend to install at least
```bash
pip install 'baybe[chem,simulation]'
```

The available groups are:
- `extras`: Installs all dependencies required for optional features.
- `benchmarking`: Required for running the benchmarking module.
- `chem`: Cheminformatics utilities (e.g. for the `SubstanceParameter`).
- `docs`: Required for creating the documentation.
- `examples`: Required for running the examples/streamlit.
- `lint`: Required for linting and formatting.
- `mypy`: Required for static type checking.
- `onnx`: Required for using custom surrogate models in [ONNX format](https://onnx.ai).
- `polars`: Required for optimized search space construction via [Polars](https://docs.pola.rs/).
- `insights`: Required for built-in model and campaign analysis (e.g. using [SHAP](https://shap.readthedocs.io/)).
- `simulation`: Enabling the [simulation](https://emdgroup.github.io/baybe/stable/_autosummary/baybe.simulation.html) module.
- `test`: Required for running the tests.
- `dev`: All of the above plus dev tools. For code contributors.

## 📡 Telemetry
Telemetry was fully and permanently removed in version 0.14.0.

## 📖 Citation
If you find BayBE useful, please consider citing [our paper](https://doi.org/10.1039/D5DD00050E):

```bibtex
@article{baybe_2025,
  author = "Fitzner, Martin and {\v S}o{\v s}i{\'c}, Adrian and 
            Hopp, Alexander V. and M{\"u}ller, Marcel and Rihana, Rim and 
            Hrovatin, Karin and Liebig, Fabian and Winkel, Mathias and 
            Halter, Wolfgang and Brandenburg, Jan Gerit",
  title  = "{BayBE}: a {B}ayesian {B}ack {E}nd for experimental planning 
            in the low-to-no-data regime",
  journal = "Digital Discovery",
  year = "2025",
  volume = "4",
  issue = "8",
  pages = "1991-2000",
  publisher = "RSC",
  doi = "10.1039/D5DD00050E",
  url = "http://dx.doi.org/10.1039/D5DD00050E",
}
```

## 🛠️ Known Issues
A list of known issues can be found [here](https://emdgroup.github.io/baybe/stable/known_issues.html).

## 👨🏻‍🔧 Maintainers

- Martin Fitzner (Merck KGaA, Darmstadt, Germany), [Contact](mailto:martin.fitzner@merckgroup.com), [Github](https://github.com/Scienfitz)
- Adrian Šošić (Merck Life Science KGaA, Darmstadt, Germany), [Contact](mailto:adrian.sosic@merckgroup.com), [Github](https://github.com/AdrianSosic)
- Alexander Hopp (Merck KGaA, Darmstadt, Germany) [Contact](mailto:alexander.hopp@merckgroup.com), [Github](https://github.com/AVHopp)

## 🙏 Contributors

Thanks to our contributors!

[![Contributors](https://contrib.rocks/image?repo=emdgroup/baybe)](https://github.com/emdgroup/baybe/graphs/contributors)

## 📄 License

Copyright 2022-2025 Merck KGaA, Darmstadt, Germany
and/or its affiliates. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
