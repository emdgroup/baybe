# Simulation

BayBE offers multiple functionalities to "simulate" experimental campaigns with a given lookup mechanism. This user guide briefly introduces how to use the methods available in our [simulation submodule](baybe.simulation).

For a wide variety of applications of this functionality, we refer to the corresponding [examples](../../examples/Backtesting/Backtesting).

## Terminology: What do we mean by "Simulation"?

The term "simulation" can have two slightly different interpretations, depending on the applied context.

1. It can refer to "backtesting" a particular experimental campaign on a fixed finite dataset.
Thus, "simulation" means investigating what experimental trajectory we would have observed if we had used different setups or recommenders and restricted the possible parameter configurations to those contained in the dataset.

2. It can refer to the simulation of an *actual* DOE loop, i.e., recommending experiments and retrieving the corresponding measurements, where the loop closure is realized in the form of a callable (black-box) function that can be queried during the optimization to provide target values. Such a callable could for instance be a simple analytical function or a numerical solver of a set of differential equations that describe a physical system.

## The Lookup Functionality

In BayBE, the simulation submodule allows a wide range of use cases and can even be used for "oracle predictions".
This is enabled by the proper use of the `lookup` functionality, which allows to either use fixed data sets, analytical functions, and general callbacks for retrieving target function values. 

All functions require a `lookup` which is used to close the loop and return target values for points in the search space.
It can be provided in the form of a dataframe or a `Callable`.

```{note}
Technically, the `lookup` can also be `None`. This results in the simulation producing random results which is not discussed further.
```

### Using a Dataframe

When choosing a dataframe, it needs to contain parameter combinations and their target results.
To make sure that the backtest produces a realistic assessment of the performance, all possible parameter combinations should be measured and present in the dataframe.
However, this is an unrealistic assumption for most applications as it is typically not the case that all possible parameter combinations have been measured prior to the optimization.
As a consequence, it might well be the case that a provided dataframe contains the measurements of only some parameter configurations while a majority of combinations is not present.
For this case, BayBE offers different ways of handling such "missing" values. 
This behavior is configured using the `impute_mode` keyword and provides the following possible choices:
- ``"error"``: An error will be thrown.
- ``"worst"``: Imputation uses the worst available value for each target.
- ``"best"``: Imputation uses the best available value for each target.
- ``"mean"``: Imputation uses the mean value for each target.
- ``"random"``: A random row will be used as lookup.
- ``"ignore"``: The search space is stripped before recommendations are made so that unmeasured experiments will not be recommended.

### Using a `Callable`

The `Callable` needs to return the target values for any given parameter combination. The only requirement that BayBE imposes on using a `Callable` as a lookup mechanism is thus that it returns either a float or a tuple of floats and to accept an arbitrary number of floats as input.

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

This function returns a dataframe that contains the results. For details on the columns of this dataframe as well as the dataframes returned by the other functions discussed here, we refer to the documentation of the submodule [here](baybe.simulation).

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