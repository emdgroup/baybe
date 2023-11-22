# Campaigns

## General information

[`Campaigns`](baybe.campaign.Campaign) are the central objects of BayBE. They are used to organize the different aspects and are the main point of interaction via python.

Campaigns define and record an experimentation process, i.e. the execution of a
series of measurements and the iterative sequence of events involved.

In particular, a campaign:
* Defines the search space over which the experimental parameter may vary via the [`Searchspace`](baybe.searchspace.core.SearchSpace) class.
* Defines the objective of an experimentation process via the [`Objective`](baybe.objective.Objective) class.
* Defines a strategy for traversing the search space via the [`Strategy`](baybe.strategies.base.Strategy) class.
* Records the measurement data collected during the process.
* Records metadata about the progress of the experimentation process.

## Attributes and properties

To construct a new campaign, it is necessary to provide two objects.
1. **A search space ``searchspace``:** The parameter space in which the campaign operates. It can be a purely discrete, purely continuous or hybir space. It is 
We refer to the [`Searchspace`](baybe.searchspace.core.SearchSpace) class resp. the corresponding [user guide on search spaces](./searchspace) for more details.
2. **An objective ``objective``:** The optimization objective. It is possible to either optimize a single target or to combine different targets. We refer to the [`Objective`](baybe.objective.Objective) class resp. the corresponding [user guide on objectives](./objectives) for more details.

The following additional aspects (and corresponding public ``arguments``) are available to further specify a campaign:
* **A strategy ``strategy``:**: The strategy that is used during the campaign. As a default, a [`TwoPhaseStrategy`](baybe.strategies.composite.TwoPhaseStrategy) is employed. For more details on strategies, see [here](./strategy).
* **Conducted experiments ``measurements_exp``**: A pandas ``DataFrame`` containing the experimental representation of previously conducted experiments. Preferably, the entries of this ``DataFrame`` were created by a previous campaign using [the campaigns recommend function](baybe.campaign.Campaign.recommend). If no such ``DataFrame`` is provided, it is assumed that no experiments were conducted previously.
* **Numerical tolerance ``numerical_measurements_must_be_within_tolerance``**: This is a flag for forcing numerical measurements to be within a pre-defined tolerance. The default value is ``True``.
* **Previously done batches ``n_batches_done``** The number of previously processed batches. The default value is 0.
* **Previously done fits ``n_fits_done``**: The number of previously done fits. The default value is 0.

## How to use campaigns

We have a detailed example explaining how to use the ``Campaign`` object in our examples, so please see [here](./../../examples/Basics/campaign) for all necessary information.