# Campaigns

Campaigns play a crucial role in Design of Experiments (DOE), and consequently also for BayBE.
They serve as a structured framework for defining and documenting an experimentation
process.

The [`Campaign`](baybe.campaign.Campaign) class is used to model campaigns and 
serves as the primary interface for interacting with BayBE as a user. This class is
responsible for handling the data, making recommendations, adding measurements and
most other tasks.

## Creating a campaign

### Using the campaign class

When constructing a campaign, it is necessary to provide two objects:
A search space (see [class](baybe.searchspace.core.SearchSpace) resp. [user guide](./searchspace))
and an optimization objective (see [class](baybe.objective.Objective) resp. [user guide](./objective)).
Optionally, it is possible to provide a different strategy
(see [class](baybe.strategies.base.Strategy) resp. [user guide](./strategy)) and specify
other aspects of the campaign (see [here](#AM) for details on
`numerical_measurements_must_be_within_tolerance`).


~~~python
from baybe import Campaign

campaign = Campaign(
    searchspace=searchspace,  # Required
    objective=objective,  # Required
    strategy=strategy,  # Optional
    numerical_measurements_must_be_within_tolerance=boolean,  # Optional
)
~~~

```{attention}
Note that we currently also expose other fields via the constructor. This is
only temporary, and the corresponding fields should be ignored.
```

### Creating a campaign via a JSON config

It is also possible to specify a `Campaign` via a configuration string and using the
function [`Campaign.from_config`](baybe.campaign.Campaign.from_config).
The specification of config files as well as what they contain are automatically
derived from the class structure.
Furthermore, a config file can be validated using 
[Campaign.validate_config](baybe.campaign.Campaign.validate_config).
For more details and a full exemplary config, we refer to the corresponding
[example](./../../examples/Serialization/create_from_config).

## Getting recommendations

### Basics

```{attention}
Obtaining recommendations resp. adding measurements using the `recommend` resp.
`add_measurements` functions is the only safe way to inform a `Campaign` object about
new measurements. These functions update the necessary metadata that is crucial for the
proper execution of a campaign. It is important to rely on these functions to maintain
the integrity and reliability of the campaign's execution.
```

To obtain a recommendation for the next experiment, we can query the campaign and use
the [`recommend`](baybe.campaign.Campaign.recommend) function. The function takes only
the `batch_quantity` keyword, specifying the desired size of
the batch of experiments to be conducted.

~~~python
rec = campaign.recommend(batch_quantity=3)
~~~

The `recommend` function returns a `DataFrame` with `batch_quantity` many rows, each 
representing a set of parameters from the search space.

```{important}
There is a difference between performing multiple recommendations
with batch size of 1 and a single recommendation with a larger batch size.
* **Batch size larger than 1**: The recommended experiments
are chosen to *jointly* optimized the acquisition function.
This means that the recommendations are made considering the interaction of multiple
experiments together.
* **Batch size of 1**: When making several smaller recommendations instead, each
*individual* recommendation optimizes the acquisition function at the specific point in 
time when it is requested. In this case, the recommendations are made independently of
each other without considering the joint optimization.

Note that this distinction might not be relevant or applicable for all possible
situation, for example when using recommenders that do not perform joint optimization.
```

### Caching of recommendations

Whenever recommendations are made, the `Campaign` object caches them. If any new 
measurements are added, then the cached recommendations are deleted. However,
if no measurements are added and the `recommend` function is called again, then the
`Campaign` object returns the cached recommendations instead of generating new
ones. In addition, the cache is also reset if the batch size of the repeated call has
changed compared to the first one. This is due to the way the batch size influences
which points are being recommended.
This caching mechanism helps to optimize performance by avoiding unnecessary
re-computations when measurements are not provided.

(AM)=
## Adding measurements

Measurements are added by expanding the  `DataFrame` that was created by the `recommend`
function by adding a new column for the target. 

~~~python
rec["Target_max"] = [2, 4, 9]  # 3 values matching the batch_quantity of 3
campaign.add_measurements(rec)
new_rec = campaign.recommend(batch_quantity=5)
~~~

For discrete parameters, measurements are required to fall into a
predefined tolerance by default.
This tolerance is defined on the level of the individual parameters.
This requirement can be disabled upon initialization of a campaign using the
`numerical_measurements_must_be_within_tolerance` flag.

## Serialization

Like most of the objects managed by BayBE, `Campaign` objects can be serialized and
deserialized using the [`to_json`](baybe.utils.serialization.SerialMixin.to_json) and
[`from_json`](baybe.utils.serialization.SerialMixin.from_json) methods.
These methods convert the `Campaign` to a string in `json` format resp. convert a string
in`json` format to a `Campaign` object. As expected,
serializing and de-serializing a campaign yields the exact identical object:
~~~python
campaign_json = campaign.to_json()
recreated_campaign = Campaign.from_json(campaign_json)
assert campaign == recreated_campaign
~~~
For more information on serialization, using the `to_json` and `from_json` methods, we
refer to the corresponding [examples](./../../examples/Serialization/Serialization).

## Further information

In all of the [examples](./../../examples/examples), a campaign is created. For more
details on how to create campaigns for a specific use case, we thus propose to have
a look at the most suitable example.