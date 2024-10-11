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
For technical reasons, not every recommender is able to utilize `pending_experiments` in
the same way. For instance,
[`BotorchRecommender`](baybe.recommenders.pure.bayesian.botorch.BotorchRecommender)
takes all pending experiments into account, even if they do not match exactly with any
point in the search space. 
[Non-predictive recommenders](baybe.recommenders.pure.nonpredictive.base.NonPredictiveRecommender) like 
[`SKLearnClusteringRecommender`](baybe.recommenders.pure.nonpredictive.clustering.SKLearnClusteringRecommender)s,
[`RandomRecommender`](baybe.recommenders.pure.nonpredictive.sampling.RandomRecommender) or
[`FPSRecommender`](baybe.recommenders.pure.nonpredictive.sampling.FPSRecommender)
only take pending points into consideration if the recommender flag
[allow_recommending_pending_experiments](baybe.recommenders.pure.nonpredictive.base.NonPredictiveRecommender.allow_recommending_pending_experiments)
is set to `False`. In that case, the candidate space is stripped of pending experiments
that are exact matches with the search space, i.e. they will not even be considered.
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
This functionality is under development as part of multi-target models.