# Campaigns

## General information

[`Campaigns`](baybe.campaign.Campaign) are the central objects of BayBE. They are used to organize the different aspects and are the main point of interaction via python.

Campaigns define and record an experimentation process, i.e. the execution of a
series of measurements and the iterative sequence of events involved.

## Attributes and properties

To construct a new campaign, it is necessary to provide two objects.
1. **A search space:** The parameter space in which the campaign operates. It can be a purely discrete, purely continuous or hybir space.
We refer to the [`Searchspace`](baybe.searchspace.core.SearchSpace) class resp. the corresponding [user guide](./searchspace) for more details.
2. **An objective:** The optimization objective. It is possible to either optimize a single target or to combine different targets. We refer to the [`Objective`](baybe.objective.Objective) class resp. the corresponding [user guide](./objective) for more details.

The following additional aspects are available to further specify a campaign:
* **A strategy:**: The strategy that is used during the campaign. As a default, a [`TwoPhaseStrategy`](baybe.strategies.composite.TwoPhaseStrategy) is employed. For more details on strategies, see [here](./strategy).
* **Conducted experiments**: A pandas ``DataFrame`` containing the experimental representation of previously conducted experiments. Preferably, the entries of this ``DataFrame`` were created by a previous campaign using [the campaigns recommend function](baybe.campaign.Campaign.recommend). If no such ``DataFrame`` is provided, it is assumed that no experiments were conducted previously.
* **Numerical tolerance**: This is a flag for forcing numerical measurements to be within a pre-defined tolerance. Note that the setting of the tolerances is controlled as a part of the respective parameter.
* **Previously done batches and fits** In case that a campaign builds upon previously condiucted experiments, it is possible to provide the number of previously done batches and fits.

## How to use campaigns

We have a detailed example explaining how to use the ``Campaign`` object in our examples, so please see [here](./../../examples/Basics/campaign) for all necessary information.