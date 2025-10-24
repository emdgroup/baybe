# FAQ

```{dropdown} Do I need to create a campaign to get recommendations?

No, creating a campaign is not mandatory.
BayBE offers two entry points for generating recommendations:
* a stateful [`Campaign.recommend`](baybe.campaign.Campaign.recommend) method and 
* a stateless [`RecommenderProtocol.recommend`](baybe.recommenders.base.RecommenderProtocol.recommend) method.

For more details on when to choose one method over the other,
see [here](userguide/getting_recommendations).
```

```{dropdown} BayBE recommends A but experimentalists do B. What now?

Don't panic and grab your towel. Recommendations from BayBE are just ... well,
"recommendations". The measurements you feed back to BayBE need not to be related to
the original recommendation in any way. In fact, requesting recommendations and adding
data are two separate actions, and there is no formal requirement to perform these
actions in any particular order nor to "respond" to recommendations in any form.

Note, however, that subsequent recommendations **may** be affected by earlier steps in
your campaign, depending on your settings for the
{attr}`~baybe.campaign.Campaign.allow_recommending_already_measured` and
{attr}`~baybe.campaign.Campaign.allow_recommending_already_recommended` flags.
```

## Checklist for designing BayBE experiments

### Defining targets

- Should be target value maximized rather than minimized or matched to specific value?

  > Specify this when
[defining the target](https://emdgroup.github.io/baybe/stable/userguide/targets.html#numericaltarget).

- Should be multiple target optimized simultaneously?

  > See how to use 
[multi-target objectives](https://emdgroup.github.io/baybe/stable/userguide/objectives.html).

### Defining parameter search space

- Are only some parameter values of interest/possible?

  > See how to exclude some 
[parameter values](https://emdgroup.github.io/baybe/stable/userguide/getting_recommendations.html#excluding-configurations) 
from being recommended, such as by defining
[bounds for continuous parameters](https://emdgroup.github.io/baybe/stable/userguide/parameters.html#numericalcontinuousparameter)
or [active values for discrete parameters](https://emdgroup.github.io/baybe/stable/userguide/parameters.html#discrete-parameters).

- Are only some parameter combinations of interest/possible?

  > See how to exclude some 
parameter combinations from being considered by using 
[constraints](https://emdgroup.github.io/baybe/stable/userguide/constraints.html) or
[constrained searchspaces](https://emdgroup.github.io/baybe/stable/userguide/searchspace.html#creating-a-simplex-bound-discrete-subspace).
Alternatively, if the aim is to use only a few specific parameter configurations the search space can be created from a 
[dataframe](https://emdgroup.github.io/baybe/stable/userguide/searchspace.html#id3) 
rather than from the product of all possible parameter combinations.

- Are some parameters non-numeric or allow only discrete numbers?

  > Use [discrete](https://emdgroup.github.io/baybe/stable/userguide/parameters.html#discrete-parameters)
rather than [continuous](https://emdgroup.github.io/baybe/stable/userguide/parameters.html#continuous-parameters) parameters.

- Is it possible to encode discrete parameters based on domain knowledge 
to capture relationships between categories
(e.g., ordered values, molecular fingerprints, model embeddings)?

  > See how to 
[encode](https://emdgroup.github.io/baybe/stable/userguide/parameters.html#discrete-parameters)
discrete parameters or provide custom encodings.

### Account for specifics of data availability or acquisition procedure

- Is additional data from historic or other partially-related experiments available?

  > Use [transfer learning](https://emdgroup.github.io/baybe/stable/userguide/transfer_learning.html).

- Will the outcome measurements of different parameter setting become available at different times?

  > Use [asynchronous workflows](https://emdgroup.github.io/baybe/stable/userguide/async.html).

### Advanced: adjust how recommendations are prioritized

- Is the aim to reduce the overall uncertainty across different regions of the search space 
rather than optimize a specific objective?

  > Use [active learning](https://emdgroup.github.io/baybe/stable/userguide/active_learning.html).




