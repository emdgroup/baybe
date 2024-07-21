# Asynchronous Workflows

Asynchronous workflows describe situations where the loop between measurements and
recommendations is more complex and needs to incorporate various other aspects. These
could for instance be:
- **Distributed workflows**: When recommendations are distributed across several
  operators, e.g. at different locations or in several reactors, some experiments might
  have been started, but are not ready when the next recommendation is requested.
  Without further consideration, the algorithm would be likely to recommend the pending
  measurements again, as it is unaware they were already started.
- **Partial targets**: Relevant when dealing with multiple targets which require very
  different amounts of time to measure. When a new recommendation is requested, the
  targets of previously recommended points might only partially available, but that
  info should ideally already be considered.


## Pending Measurements

With *pending measurements* we mean recommendations whose measurement process has
been started, but not yet completed by time of triggering the next set of
recommendations. This is not to be confused with a situation where the targets of a
recommendation have been partially measured. A measurement is *pending* whenever one or
more of its targets are not measured yet. Only once all the targets are available,
the data point can be removed from `pending_measurements` and be provided via 
`measurements`.

To avoid repeated recommendations in the above scenario, BayBE provides the 
`pending_measurements` keyword. It is available in `.recommend` from
[`Campaign`](baybe.campaign.Campaign) or any
[`Recommender`](baybe.recommenders.base.RecommenderProtocol).

```{admonition} Supported Acquisition Functions
:class: important
`pending_measurementes` is only supported by Monte Carlo acquisition functions, i.e. the
ones that start with a `q` in their name. Attempting to use a non-MC acquisition
function with `pending_measurements` will result in a 
[`IncompatibleAcquisitionFunctionError`](baybe.exceptions.IncompatibleAcquisitionFunctionError).
```

```{admonition} Supported Recommenders
:class: important
For technical reasons, not every recommender might be able to fully or even partially
utilize `pending_measurements`. For instance, 
[`BotorchRecommender`](baybe.recommenders.pure.bayesian.botorch.BotorchRecommender)
takes all pending measurements into account, even if they do not match exactly with any
point in the searchspace.
[Clustering recommenders](baybe.recommenders.pure.nonpredictive.clustering.SKLearnClusteringRecommender)
only incorporate pending points which are exact searchspace matches by excluding them
from the candidates. The 
[`RandomRecommender`](baybe.recommenders.pure.nonpredictive.sampling.RandomRecommender)
does not take pending points into consideration at all and will raise a correspoding
[warning](baybe.exceptions.UnusedObjectWarning).
```

Akin to `measurements` or `recommendations`, `pending_measurements` is a dataframe in
experimental representation. In the following example, we get a set of recommendations,
add results for half of them, and start the next recommendation, marking the
other half pending:
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