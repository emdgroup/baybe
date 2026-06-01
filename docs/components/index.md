# Components

BayBE follows a modular design in which individual components can be independently
crafted and composed together to build a complete optimization workflow.
This modularity is one of BayBE's core strengths, offering two key advantages:

1. Components are configured on a user-friendly level: low-level implementation
   details are hidden by default but remain accessible when needed.
2. Components can easily be swapped against alternatives in a plug-and-play fashion,
   making it straightforward to compare different setups.

The purpose of this user guide is to explain these components and their interactions.

While advanced users can leverage this flexibility to fine-tune every aspect of their
workflow, newcomers will most likely start with the {class}`~baybe.campaign.Campaign` as
their first point of interaction. Campaigns are high-level objects that allow you to
define a particular optimization problem, suggest new measurements, and administer the
current state of your experimental operation. The diagram below shows how a campaign
is built from other components and integrates into the Bayesian optimization loop.

```{image} ../_static/api_overview_dark.svg
:align: center
:class: only-dark
```

```{image} ../_static/api_overview_light.svg
:align: center
:class: only-light
```

```{toctree}
:maxdepth: 2

Campaigns <campaigns>
Constraints <constraints>
Insights <insights>
Objectives <objectives>
Parameters <parameters>
Recommenders <recommenders>
Search Spaces <searchspace>
Surrogates <surrogates>
Targets <targets>
Transformations <transformations>
Utilities <utils>
```
