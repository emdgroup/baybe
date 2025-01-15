# FAQ

**Do I need to create a campaign to get recommendations?**

No, creating a campaign is not mandatory.
BayBE offers two entry points for generating recommendations:
* a stateful [`Campaign.recommend`](baybe.campaign.Campaign.recommend) method and 
* a stateless [`RecommenderProtocol.recommend`](baybe.recommenders.base.RecommenderProtocol.recommend) method.

For more details on when to choose one method over the other,
see [here](userguide/getting_recommendations).
