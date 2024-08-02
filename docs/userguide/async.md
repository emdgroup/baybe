# Asynchronous Workflows

Asynchronous workflows describe situations where the loop between measurement and
recommendation is more complex and needs to incorporate various other aspects. These
could for instance be:
- **Distributed workflows**: When recommendations are distributed across several
  operators, e.g. at different locations or in several reactors, some experiments might
  have been started, but are not ready when the next batch of recommendations is requested.
  Without further consideration, the algorithm would be likely to recommend the pending
  measurements again, as it is unaware they were already started.
- **Partial targets**: When dealing with multiple targets that require very different
  amounts of time to measure, the targets of previously recommended points might only be
  partially available when requesting the next batch of recommendations. Still, these
  partial measurements should ideally be considered when generating the recommendations.


## Pending Measurements

With *pending measurements* we mean recommendations whose measurement process has
been started, but not yet completed by time of triggering the next set of
recommendations. This is not to be confused with a situation where the targets of a
recommendation have been partially measured. A measurement is *pending* whenever one or
more of its targets are not measured yet. Only once all the targets are available,
the data point can be removed from `pending_measurements` and be provided via 
`measurements`.

To avoid repeated recommendations in the above scenario, BayBE provides the 
`pending_measurements` keyword. It is available wherever recommendations can be
requested, i.e. [`Campaign.recommend`](baybe.campaign.Campaign.recommend) or 
[`RecommenderProtocol.recommend`](baybe.recommenders.base.RecommenderProtocol.recommend).

```{admonition} Supported Acquisition Functions
:class: important
`pending_measurementes` is only supported by Monte Carlo (MC) acquisition functions, i.e. the
ones that start with a `q` in their name. Attempting to use a non-MC acquisition
function with `pending_measurements` will result in an 
[`IncompatibleAcquisitionFunctionError`](baybe.exceptions.IncompatibleAcquisitionFunctionError).
```

```{admonition} Supported Recommenders
:class: important
For technical reasons, not every recommender is able to fully or even partially
utilize `pending_measurements`. For instance, 
[`BotorchRecommender`](baybe.recommenders.pure.bayesian.botorch.BotorchRecommender)
takes all pending measurements into account, even if they do not match exactly with any
point in the search space.
[Clustering recommenders](baybe.recommenders.pure.nonpredictive.clustering.SKLearnClusteringRecommender)
only incorporate pending points that are exact search space matches by excluding them
from the candidates. The 
[`RandomRecommender`](baybe.recommenders.pure.nonpredictive.sampling.RandomRecommender)
does not take pending points into consideration at all and will raise a correspoding
[warning](baybe.exceptions.UnusedObjectWarning).
```

Akin to `measurements` or `recommendations`, `pending_measurements` is a dataframe in
experimental representation (see [here](/userguide/searchspace)). 
In the following example, we get a set of recommendations, add results for half of them,
and start the next recommendation, marking the other half pending:
```python
# Get a set of 10 recommendation
rec = campaign.recommend(batch_size=10)

# Split recommendations into two parts
rec_finished = rec.iloc[:5]
rec_pending = rec.iloc[5:]

# Add target measurements to the finished part. Here we add a random number
rec_finished["Target_max"] = 1337
campaign.add_measurements(rec_finished)

# Get the next set of recommendations, incorporating the still unfinished ones
# These will not include the pending recommendations again
rec_next = campaign.recommend(10, pending_measurements=rec_pending)
```