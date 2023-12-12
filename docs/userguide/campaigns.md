# Campaigns

Campaigns play a crucial role in Design of Experiments, and consequently also for BayBE.
They serve as a structured framework for defining and documenting an experimentation
process. The [`Campaign`](baybe.campaign.Campaign) class is used to model campaigns and 
serves as the primary interface for interacting with BayBE.
While creating a campaign can be as simple as specifying a search space with experimental
parameters and an optimization objective, there are numerous additional aspects that can
be customized and fine-tuned.

## Our example: Optimizing the brewing of coffee

Since a researcher is a person turning caffeine into either theory or code, we use the
example of optimizing the process of brewing coffee in this user guide.

We use the following parameters for optimizing our brewing process:
* **Grind size:** The grind size influences the extraction of the flavors.
The available options are ``[coarse, rather_coarse, rather_fine, fine]``.
* **Water temperature:** The temperature affects the rate at which the coffee compounds
are extracted. We can vary the temperature in 0.1°C increments within the range of
90.5°C to 96°C.
* **Brewing time:** The brewing time can influence the coffee flavor due to potential
over- or under-extraction. We allow brewing times between 120 and 360 seconds, with the
ability to stop brewing after each 10-second interval.

## Minimal definition of a campaign

When constructing a campaign, it is necessary to provide two objects:
A search space (see [class](baybe.searchspace.core.SearchSpace) resp. [user guide](./searchspace))
and an optimization objective (see [class](baybe.objective.Objective) resp. [user guide](./objective)).

The search space describes the possible configuration of parameters that can be tested during the campaign.

```python
from baybe.parameters import CategoricalParameter, NumericalDiscreteParameter
from baybe.searchspace import SearchSpace

grind_size = CategoricalParameter(name="grind_size", values=["coarse", "rather_coarse", "rather_fine", "fine"])
temp = NumericalDiscreteParameter(name="temp", values = tuple([t/10.0 for t in range (905, 961)]))
time = NumericalDiscreteParameter(name="time", values = tuple(range(120, 370, 10)))

space = SearchSpace.from_product([grind_size, temp, time])
```

We refer to the [parameters](./parameters) resp. [search spaces](./searchspace) user
guides for more details.

To assess the quality of our coffee, we use a subjective evaluation method
involving tasting and assigning a rating on a scale of 1 to 10, where a higher score
indicates better quality.

```python
from baybe.targets import NumericalTarget
from baybe.objective import Objective

target = NumericalTarget(name="quality", mode="MAX", bounds=(1,10))
objective = Objective(mode="SINGLE", targets=[target])
```

Combining the search space and the objective now enables us to create a campaign.

```python
from baybe.campaign import Campaign

campaign = Campaign(searchspace=space, objective=objective)
```

## Getting a recommendation and adding a measurement

```{attention}
Adding recommendations and measurements using the `recommend` and `add_measurements`
functions is the safe only way to inform a `Campaign` object about new measurements.
These functions update the necessary metadata that is crucial for the proper
execution of a campaign. It is important to rely on these functions to maintain the
integrity and reliability of the campaign's execution.
```

To obtain a recommendation for the next experiment, we can query the campaign and use
the [`recommend`](baybe.campaign.Campaign.recommend) function. The function takes only
one argument, which is the `batch_quantity` keyword. This specifies the desired size of
the batch of experiments to be conducted.

```python
rec = campaign.recommend(batch_quantity=3)
```

The `recommend` function returns a `DataFrame` with `batch_quantity` many rows, each 
representing a set of parameters from the search space.
To add measurements, we expand the `DataFrame` by adding a new column for the target.
We can then provide the campaign with these measurements and receive a new
recommendation.

```python
rec["quality"] = [2,4,9]
campaign.add_measurements(rec)
new_rec = campaign.recommend(batch_quantity=5)
```

## Further specification of campaigns

Although only a search space and an objective are necessary to create a campaign,
several other aspects can be changed by the user.
1. **A strategy:**: By default, campaigns use the composite
[`TwoPhaseStrategy`](baybe.strategies.composite.TwoPhaseStrategy).
This can be changed using the `strategy` keyword.
2. **Numerical tolerance**: By default, numerical measurements are required to fall into
a predefined tolerance. This requirement can be disabled by using the
`numerical_measurements_must_be_within_tolerance` flag.


## Details on design and functionality

### Batch sizes and their influence on recommendations

The `batch_quantity` keyword allows you to adjust the number of recommendations returned
by the `recommend` function. However, it is important to understand the difference
between performing multiple recommendations with batch size of 1 and a single
recommendation with a larger batch size.
* **Larger batch size**: When using a larger batch size, the recommended experiments are
chosen to *jointly* optimized the acquisition function.
This means that the recommendations are made considering the interaction of multiple
experiments together.
* **Smaller batch size**: When making several smaller recommendations, each *individual*
recommendation optimizes the acquisition function at the specific point in time when it
is requested. In this case, the recommendations are made independently of each other
without considering the joint optimization.

### Caching of recommendations

Whenever recommendations are made, the `Campaign` object caches them. If measurements
for the recommendations are added, then the cached recommendations are deleted. However,
if no measurements are added and the `recommend` function is called again, then the
`Campaign` object simply returns the cached recommendations instead of generating new
ones. This caching mechanism helps to optimize performance by avoiding unnecessary
re-computations when measurements are not provided.

### Serialization

Like most of the objects managed by BayBE, `Campaign` objects in BayBE can be serialized
and deserialized using the [`to_json`](baybe.utils.serialization.SerialMixin.to_json)
method. This method converts the `Campaign` to a string in `json` format. As expected,
serializing and de-serializing a campaign yields the exact identical object:
```python
campaign_json = campaign.to_json()
recreated_campaign = Campaign.from_json(campaign_json)
assert campaign == recreated_campaign
```
For more information on serialization, using the `to_json` and `from_json` methods, we
refer to the corresponding [examples](./../../examples/Serialization/Serialization).

It is also possible to specify a `Campaign` via a configuration string and the function
[`Campaign.from_config`](baybe.campaign.Campaign.from_config).
As fully specifying a configuration takes too much, we refer to the corresponding
[example](./../../examples/Serialization/create_from_config).

### Further information

For an additional and more condensed example explaining the `Campaign` object, we refer
to the corresponding [example](./../../examples/Basics/campaign).