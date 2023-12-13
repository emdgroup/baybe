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

## Further specification of campaign

Although only a search space and an objective are necessary to create a campaign,
several other aspects can be changed by the user.
1. **A strategy:**: By default, campaigns use the composite
[`TwoPhaseStrategy`](baybe.strategies.composite.TwoPhaseStrategy). This strategy uses a
[`RandomRecommender`](baybe.recommenders.sampling.RandomRecommender) for the initial
recommendation and then switches to a [`SequentialGreedyRecommender`](baybe.recommenders.bayesian.SequentialGreedyRecommender).
However, using the `strategy` keyword, you can specify a different strategy.
For more details on strategies, see the corresponding [`submodule`](baybe.strategies) or
[user guide](./strategy).
* **Previously conducted experiments**: If you have previously conducted experiments and
have a `DataFrame` representing them, you can provide it using the `measurements_exp`
keyword. Preferably, the `DataFrame` was created by a previous
campaign using [the campaigns recommend function](baybe.campaign.Campaign.recommend).
If no `DataFrame` is provided, it is assumed that no experiments were conducted previously.
* **Numerical tolerance**: By default, numerical measurements are required to fall into a
predefined tolerance. However, you can disable this requirement by using the
`numerical_measurements_must_be_within_tolerance` flag.
Note that the setting of the tolerances is controlled as a part of the respective
parameter
* **Previously done batches and fits** If the campaign builds upon previously
conducted experiments, it is possible to provide the number of previously done batches
and fits. This is done via the keywords `n_batches_done` resp. `n_fits_done`.

## Further information

For an additional and more condensed example explaining the `Campaign` object, we refer
to the corresponding [example](./../../examples/Basics/campaign).