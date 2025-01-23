# Getting Recommendations

The core functionality of BayBE is its ability to generate context-aware recommendations
for your experiments. This page covers the basics of the corresponding user interface,
assuming that a {class}`~baybe.searchspace.core.SearchSpace` object and optional
{class}`~baybe.objectives.base.Objective` and measurement objects are already in place
(for more details, see the corresponding [search space](/userguide/searchspace) /
[objective](/userguide/objectives) user guides).


## The `recommend` Call

BayBE offers two entry points for requesting recommendations:
* (stateless)= 
  **Recommenders**\
  If a single (batch) recommendation is all you need, the most direct way to interact is
  to ask one of BayBE's recommenders for it, by calling its
  {meth}`~baybe.recommenders.base.RecommenderProtocol.recommend` method. To do so,
  simply pass all context information to the method call. This way, you interact with
  BayBE in a completely *stateless* way since all relevant components are explicitly
  provided at call time.

  ```{admonition} Meta Recommenders
  :class: caution

  A notable exception are {class}`~baybe.recommenders.meta.base.MetaRecommender`s
  which, depending on their configuration, may be stateless only with respect
  to the recommendation *context* but not the recommendation *mechanism*. More
  specifically, meta recommenders may – by design – generate different recommendations
  when confronted with an otherwise identical context twice, as indicated by their
  {attr}`~baybe.recommenders.meta.base.MetaRecommender.is_stateful` property. 
  ```

  For example, using the {class}`~baybe.recommenders.pure.bayesian.botorch.BotorchRecommender`:
  ```python
  recommender = BotorchRecommender()
  recommendation = recommender.recommend(batch_size, searchspace, objective, measurements)
  ```
* (stateful)= 
  **Campaigns**\
  By contrast, if you plan to run an extended series of experiments where you feed newly
  arriving measurements back to BayBE and ask for a refined experimental design,
  creating a {class}`~baybe.campaign.Campaign` object that tracks the experimentation
  progress is a better choice. This offers *stateful* way of interaction where
  the context is fully maintained by the campaign object:
  ```python
  recommender = BotorchRecommender()
  campaign = Campaign(searchspace, objective, recommender)
  campaign.add_measurements(measurements)
  recommendation = campaign.recommend(batch_size)
  ```
  For more details, have a look at our [campaign user guide](/userguide/campaigns).


## Excluding Configurations

Excluding certain parameter configurations from recommendation is generally done by
adjusting the {class}`~baybe.searchspace.core.SearchSpace` object accordingly, which
defines the set of candidate configurations that will be considered. 
* **Recommenders**\
  The above means that, for [stateless queries](#stateless), you can simply pass a
  different search space object to each call according to your needs:
  ```python
  # Recommendation with full search space
  searchspace_full = CategoricalParameter("p", ["A", "B", "C"]).to_searchspace()
  recommender.recommend(batch_size, searchspace_full, objective, measurements)

  # Recommendation with reduced search space
  searchspace_reduced = CategoricalParameter("p", ["A", "B"]).to_searchspace()
  recommender.recommend(batch_size, searchspace_reduced, objective, measurements)
  ```

* **Campaigns**\
  Because the search space must be defined before a
  {class}`~baybe.campaign.Campaign` object can be created, a different approach is
  required for [stateful queries](#stateful). For this purpose,
  {class}`~baybe.campaign.Campaign`s provide a
  {meth}`~baybe.campaign.Campaign.toggle_discrete_candidates` method that allows to
  dynamically enable or disable specific candidates while the campaign is running.
  The above example thus translates to:
  ```python
  campaign = Campaign(searchspace_full, objective, measurements)
  campaign.add_measurements(measurements)

  # Recommendation with full search space
  campaign.recommend(batch_size)

  # Recommendation with reduced search space
  campaign.toggle_discrete_candidates(pd.DataFrame({"p": ["C"]}), exclude=True)
  campaign.recommend(batch_size)
  ```
  Note that you can alternatively toggle candidates by passing the appropriate
  {class}`~baybe.constraints.base.DiscreteConstraint` objects.
  For more details, see {meth}`baybe.campaign.Campaign.toggle_discrete_candidates`.

  ```{admonition} Trajectory-Based Control
  :class: seealso

  {class}`~baybe.campaign.Campaign`s allow you to further control the candidate
  generation based on the experimental trajectory taken via their `allow_*` 
  {ref}`flags <userguide/campaigns:Candidate Control in Discrete Spaces>`.
  ```

  ```{admonition} Continuous Constraints
  :class: attention

  Currently, candidate exclusion at the {class}`~baybe.campaign.Campaign` level is
  only possible for the discrete part of the underlying search space. To restrict
  the continuous part of the search, use 
  {class}`~baybe.constraints.base.ContinuousConstraint`s when creating the space.
  ```



  

