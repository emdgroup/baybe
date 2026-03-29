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

