# Simulation

BayBE offers multiple functionalities to "simulate" experimental campaigns with a given lookup mechanism. This user guide briefly introduces how to use the methods available in our [simulation subpackage](baybe.simulation).

For a wide variety of applications of this functionality, we refer to the corresponding [examples](../../examples/Backtesting/Backtesting).

## Terminology: What do we mean by "Simulation"?

The term "simulation" can have two slightly different interpretations, depending on the applied context.

1. It can refer to "backtesting" a particular experimental campaign on a fixed finite dataset.
Thus, "simulation" means investigating what experimental trajectory we would have observed if we had used different setups or recommenders and restricted the possible parameter configurations to those contained in the dataset.

2. It can refer to the simulation of an *actual* DOE loop, i.e., recommending experiments and retrieving the corresponding measurements, where the loop closure is realized in the form of a callable (black-box) function that can be queried during the optimization to provide target values. Such a callable could for instance be a simple analytical function or a numerical solver of a set of differential equations that describe a physical system.

## The Lookup Mechanism

BayBE's simulation package enables a wide range of use cases and can even be used for "oracle predictions".
This is made possible through the flexible use of lookup mechanisms, which act as the loop-closing element of an optimization loop.

Lookups can be provided in a variety of ways, by using fixed data sets, analytical functions, or any other form of black-box callable.
In all cases, their role is the same: to retrieve target values for parameter configurations suggested by the recommendation engine.

### Using a `Callable`

Using a `Callable` is the most general way to provide a lookup mechanism.
Any `Callable` is a suitable lookup as long as it accepts a dataframe containing parameter configurations and returns the corresponding target values.
More specifically:
- The input is expected to be a dataframe whose column names contain the parameter names and whose rows represent valid parameter configurations.
- The returned output must be a dataframe whose column names contain the target names and whose rows represent valid target values.
- The indices of the input and output dataframes must match.

An example might look like this:
```python
import pandas as pd

from baybe.parameters import NumericalContinuousParameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget

searchspace = SearchSpace.from_product(
    [
        NumericalContinuousParameter("p1", [0, 1]),
        NumericalContinuousParameter("p2", [-1, 1]),
    ]
)
objective = NumericalTarget("t1", "MAX").to_objective()


def lookup(df: pd.DataFrame) -> pd.DataFrame:
    """Map parameter configurations to target values."""
    return pd.DataFrame({"t1": df["p1"] ** 2}, index=df.index)


lookup(searchspace.continuous.sample_uniform(10))
```

````{admonition} Array-Based Callables
:class: tip
If you already have a lookup callable available in an array-based format (for instance,
if your lookup values are generated using third-party code that works with array inputs
and outputs), you can effortlessly convert this callable into the required
dataframe-based format by applying our
{func}`~baybe.utils.dataframe.arrays_to_dataframes` decorator. 

For example, the above lookup can be equivalently created as follows:
```python
import numpy as np

from baybe.utils.dataframe import arrays_to_dataframes


@arrays_to_dataframes(["p1"], ["t1"])
def lookup(array: np.ndarray) -> np.ndarray:
    """The same lookup function in array logic."""
    return array**2
```

````


### Using a Dataframe

When dealing with discrete search spaces, it is also possible to provide the lookup values in a tabular representation using a dataframe.
To be a valid lookup, the dataframe must have columns corresponding to all parameters and targets in the modeled domain. 

An example might look as follows:
```python
import pandas as pd

from baybe.parameters import NumericalDiscreteParameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget

searchspace = SearchSpace.from_product(
    [
        NumericalDiscreteParameter("p1", [0, 1, 2, 3]),
        NumericalDiscreteParameter("p2", [1, 10, 100, 1000]),
    ]
)
objective = NumericalTarget("t", "MAX").to_objective()

lookup = pd.DataFrame.from_records(
    [
        {"p1": 0, "p2": 100, "t": 23},
        {"p1": 2, "p2": 10, "t": 5},
        {"p1": 3, "p2": 1000, "t": 56},
    ]
)
```

```{admonition} Missing Lookup Values
:class: tip
Ideally, all possible parameter combinations should be measured and represented in the dataframe to ensure that a backtesting simulation produces a realistic assessment of performance.
However, this is an unrealistic assumption for most applications because search spaces are oftentimes exceedingly large.
As a consequence, it may well be the case that a provided dataframe contains the measurements of only some parameter configurations while the majority of combinations is not present (like in the example above).
To address this issue, BayBE provides various methods for managing these “missing” targets,
which can be configured using the {paramref}`~baybe.simulation.lookup.look_up_targets.impute_mode`
keyword of the respective simulation function.
```

### Using `None`

When testing code, it can sometimes be helpful to have an "arbitrary" lookup mechanism available without having to craft a custom one.
An example of when this is useful is when evaluating the actual lookup is too expensive and results in too long turnaround times (for instance, when the lookup is implemented by running complex code such as a computer simulation).
In these situations, using `None` as lookup can save valuable development time, which invokes the {func}`~baybe.utils.dataframe.add_fake_measurements` utility behind the scenes to generate random target values for any given domain.


## Simulating a Single Experiment

The function [`simulate_experiment`](baybe.simulation.core.simulate_experiment) is the most basic form of simulation.
It runs a single execution of a DoE loop for either a specific number of iteration or until the search space is fully observed.

For using this function, it is necessary to provide a [`campaign`](baybe.campaign.Campaign). Although technically not necessary, we advise to also always provide a lookup mechanisms since fake results will be produced if none is provided. It is possible to specify several additional parameters like the batch size, initial data or the number of DoE iterations that should be performed

~~~python
results = simulate_experiment(
    # Necessary
    campaign=campaign,
    # Technically optional but should always be set
    lookup=lookup,
    # Optional
    batch_size=batch_size,
    n_doe_iterations=n_doe_iterations,
    initial_data=initial_data,
    random_seed=random_seed,
    impute_mode=impute_mode,
    noise_percent=noise_percent,
)
~~~

This function returns a dataframe that contains the results. For details on the columns of this dataframe as well as the dataframes returned by the other functions discussed here, we refer to the documentation of the subpackage [here](baybe.simulation).

## Simulating Multiple Scenarios

The function [`simulate_scenarios`](baybe.simulation.scenarios.simulate_scenarios) allows to specify multiple simulation settings at once.
Instead of a single campaign, this function expects a dictionary of campaigns, mapping scenario identifiers to `Campaign` objects.
In addition to the keyword arguments available for `simulate_experiment`, this function has two different keywords available:
1. `n_mc_iterations`: This can be used to perform multiple Monte Carlo runs with a single call. Multiple Monte Carlo runs are always advised to average out the effect of random effects such as the initial starting data.
2. `initial_data`: This can be used to provide a list of dataframe, where each dataframe is then used as initial data for an independent run. That is, the function performs one optimization loop per dataframe in this list.

Note that these two keywords are mutually exclusive.

~~~python
lookup = ...  # some reasonable lookup, e.g. a Callable
campaign1 = Campaign(...)
campaign2 = Campaign(...)
scenarios = {"Campaign 1": campaign1, "Campaign 2": campaign2}

results = simulate_scenarios(
    scenarios=scenarios,
    lookup=lookup,
    batch_size=batch_size,
    n_doe_iterations=n_doe_iterations,
    n_mc_iterations=n_mc_iterations,
)
~~~

## Simulating Transfer Learning

The function [`simulate_transfer_learning`](baybe.simulation.transfer_learning.simulate_transfer_learning) partitions the search space into its tasks and simulates each task with the training data from the remaining tasks.

```{note}
Currently, this only supports discrete search spaces. See [`simulate_transfer_learning`](baybe.simulation.transfer_learning.simulate_transfer_learning) for the reasons.
```

~~~python
task_param = TaskParameter(
    name="Cell Line",
    values=["Liver Cell", "Brain Cell", "Skin Cell"],
)
# Define searchspace using a task parameter
searchspace = SearchSpace.from_product(parameters=[param1, param2, task_param])

# Create a suitable campaign
campaign = Campaign(searchspace=searchspace, objective=objective)

# Create a lookup dataframe. Note that this needs to have a column labeled "Function"
# with values "F1" and "F2"
lookup = DataFrame(...)

results = simulate_transfer_learning(
    campaign=campaign,
    lookup=lookup,
    batch_size=BATCH_SIZE,
    n_doe_iterations=N_DOE_ITERATIONS,
    n_mc_iterations=N_MC_ITERATIONS,
)
~~~