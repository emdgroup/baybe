# User Guide

The most commonly used interface BayBE provides is the central 
[`Campaign`](baybe.campaign.Campaign) object,
which suggests new measurements and administers the current state of 
your experimental operation. The diagram below explains how the 
[`Campaign`](baybe.campaign.Campaign) can be used to perform 
the **Bayesian Optimization Loop**, how it can be **Configured** and 
how the results can be **Post-Analysed**.
which suggests new measurements according to the bayesian optimization
procedure. The below diagram explains how the [`Campaign`](baybe.campaign.Campaign)
can be used to perform the **Bayesian Optimization Loop**, 
how the [`Campaign`](baybe.campaign.Campaign) can be **Configured** and 
how the results can be **Post-Analysed**.

```{image} ../_static/api_overview_dark.svg
:align: center
:class: only-dark
```

```{image} ../_static/api_overview_light.svg
:align: center
:class: only-light
```

Detailed examples of how to use individual API components can be found below:

```{toctree}
Getting Recommendations <getting_recommendations>
Campaigns <campaigns>
Active Learning <active_learning>
Asynchronous Workflows <async>
Constraints <constraints>
Environment Vars <envvars>
Insights <insights>
Objectives <objectives>
Parameters <parameters>
Recommenders <recommenders>
Search Spaces <searchspace>
Serialization <serialization>
Simulation <simulation>
Surrogates <surrogates>
Targets <targets>
Transfer Learning <transfer_learning>
Utilities <utils>
```