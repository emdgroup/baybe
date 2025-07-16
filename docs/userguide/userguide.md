# User Guide

BayBE is organized around a central [`Campaign`](baybe.campaign.Campaign) object,
which suggests new measurements according to the bayesian optimization
procedure. The below diagram explains how the [`Campaign`](baybe.campaign.Campaign)
can be used to perform the **Bayesian Optimization Loop**, 
how the [`Campaign`](baybe.campaign.Campaign) can be **Configured** and 
how the results can be **Post-Analysed**.

![Transforms](../_static/api_overview_light.svg)

Detailed examples of how to use individual API components can be found bellow:

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