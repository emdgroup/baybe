# Asynchronous Workflows

Asynchronous workflows describe situations where the loop between measurement and
recommendation is more complex and needs to incorporate various other aspects. These
could for instance be:
- **Distributed workflows**: When recommendations are distributed across several
  operators, e.g. at different locations or in several reactors, some experiments might
  have been started, but are not ready when the next batch of recommendations is requested.
  Without further consideration, the algorithm would be likely to recommend the pending
  experiments again (since they were and still are considered most promising), as it is
  unaware they were already started.
- **Partial targets**: When dealing with multiple targets that require very different
  amounts of time to measure, the targets of previously recommended points might only be
  partially available when requesting the next batch of recommendations. Still, these
  partial experiments should ideally be considered when generating the recommendations.

With *pending experiments* we mean experiments whose measurement process has
been started, but not yet completed by time of triggering the next set of
recommendations – this is typically the case when at least one of the configured
targets has not yet been measured.

There are two levels of dealing with such situations:
1) **Marking experiments as pending**: If an experiment is not completed (meaning at least one target is not yet measured), its
  data cannot be added as a regular measurement. However, it can be marked as pending via
  `pending_experiments` in `recommend`.
1) **Adding partial results**: If an experiment is partially completed (meaning at least one target has been
  measured), we can already update the model with the available information
  by adding a *partial* measurement.

(PENDING_EXPERIMENTS)=
## Marking Experiments as Pending

To avoid repeated recommendations in the above scenario, BayBE provides the 
`pending_experiments` keyword. It is available wherever recommendations can be
requested, i.e. [`Campaign.recommend`](baybe.campaign.Campaign.recommend) or 
[`RecommenderProtocol.recommend`](baybe.recommenders.base.RecommenderProtocol.recommend).

```{admonition} Supported Acquisition Functions
:class: important
`pending_experiments` is only supported by Monte Carlo (MC) acquisition functions, i.e. the
ones that start with a `q` in their name. Attempting to use a non-MC acquisition
function with `pending_experiments` will result in an 
[`IncompatibleAcquisitionFunctionError`](baybe.exceptions.IncompatibleAcquisitionFunctionError).
```

```{admonition} Supported Recommenders
:class: important

For technical reasons, not every recommender is able to make use of `pending_experiments`.
For instance,
[`BotorchRecommender`](baybe.recommenders.pure.bayesian.botorch.BotorchRecommender)
takes all pending experiments into account, even if they do not match exactly with 
points in the search space. 
By contrast,
[Non-predictive recommenders](baybe.recommenders.pure.nonpredictive.base.NonPredictiveRecommender) like 
[`SKLearnClusteringRecommender`](baybe.recommenders.pure.nonpredictive.clustering.SKLearnClusteringRecommender)s,
[`RandomRecommender`](baybe.recommenders.pure.nonpredictive.sampling.RandomRecommender) or
[`FPSRecommender`](baybe.recommenders.pure.nonpredictive.sampling.FPSRecommender)
do not consider `pending_experiments` at all and raise an
{class}`~baybe.exceptions.UnusedObjectWarning` when such points are passed.
```

Akin to `measurements` or `recommendations`, `pending_experiments` is a dataframe in
[experimental representation](DATA_REPRESENTATION). 
In the following example, we get a set of recommendations, add results for half of them,
and start the next recommendation, marking the other half pending:
```python
from baybe.utils.dataframe import add_fake_measurements

# Get a set of 10 recommendation
rec = campaign.recommend(batch_size=10)

# Split recommendations into two parts
rec_finished = rec.iloc[:5]
rec_pending = rec.iloc[5:]

# Add target measurements to the finished part. Here we add fake results
add_fake_measurements(rec_finished, campaign.targets)
campaign.add_measurements(rec_finished)

# Get the next set of recommendations, incorporating the still unfinished experiments.
# These will not include the experiments marked as pending again.
rec_next = campaign.recommend(10, pending_experiments=rec_pending)
```

## Adding Partial Results
A *partial result* is possible if you have multiple targets, but only measured the 
outcome for some of those. This is a common occurrence, especially if the different 
target measurements correspond to experiments that differ in complexity or duration.

As a simple example, consider a campaign with medical background aimed at creating a 
drug formulation. Typically, there are quick initial analytics performed on the 
formulation, followed by *in vitro* experiments followed by mouse *in vivo* experiments.
Without the ability to use partial measurements, you would have to wait until the slow 
mouse experiment for a given recommendation is measured until you could utilize any of 
the other (faster) experimental outcomes for that recommendation. Furthermore, if the fast 
measurements are already unpromising, the slower target measurements are possibly never 
performed at all.

In BayBE, you can leverage results even if they are only partial. This is indicated 
by setting the corresponding target measurement value to NaN. There are several ways to indicate this, e.g.:
* [`numpy.nan`](numpy.nan)
* [`pandas.NA`](pandas.NA)
* `None`
* `float("nan")`

Let us consider this 3-batch of recommendations, assuming 
we need to measure "Target_1", "Target_2" and "Target_3":
```python
import numpy as np
import pandas as pd

rec = campaign.recommend(batch_size=3)
# Resetting the index to have easier access via .loc later
measurements = rec.reset_index(drop=True)

# Add measurement results
measurements.loc[0, "Target_1"] = 10.3
measurements.loc[0, "Target_2"] = 0.5
measurements.loc[0, "Target_3"] = 11.1

measurements.loc[1, "Target_1"] = 7.1
measurements.loc[1, "Target_2"] = np.nan  # not measured yet
measurements.loc[1, "Target_3"] = 12.2

measurements.loc[2, "Target_1"] = 11.4
measurements.loc[2, "Target_2"] = pd.NA  # not measured yet
measurements.loc[2, "Target_3"] = None  # not measured yet

measurements

# Proceed with campaign.add_measurements ...
```

| Param_1 | Param_2 | ...  | Target_1 | Target_2 | Target_3 |
|:--------|:--------|------|---------:|---------:|---------:|
| on      | 1.1     | ...  |     10.3 |      0.5 |     11.1 |
| on      | 3.8     | ...  |      7.1 |      NaN |     12.2 |
| off     | 2.9     | ...  |     11.4 |      NaN |      NaN |

Internally, the incomplete rows are dropped when fitting a surrogate model for each
target. If you use an unsupported surrogate model, an error will be thrown at runtime.

````{admonition} Limitations
:class: important
The described workflow is only possible if the underlying mechanism of the used
{class}`~baybe.objectives.base.Objective` can handle missing values, as indicated
by its {attr}`~baybe.objectives.base.Objective.supports_partial_measurements` property.

For example, the
[`DesirabilityObjective`](baybe.objectives.desirability.DesirabilityObjective)
rejects partial measurement results when its
{attr}`~baybe.objectives.desirability.DesirabilityObjective.as_pre_transformation` flag
is set to `True`:
```python
from baybe.objectives import DesirabilityObjective
from baybe.targets import NumericalTarget

t1 = NumericalTarget.normalized_ramp("t1", cutoffs=(0, 1))
t2 = NumericalTarget.normalized_sigmoid("t2", anchors=[(0.0, 0.1), (1.0, 0.9)])

obj = DesirabilityObjective(targets=[t1, t2])
obj_pre = DesirabilityObjective(targets=[t1, t2], as_pre_transformation=True)
assert obj.supports_partial_measurements
assert not obj_pre.supports_partial_measurements
```
````