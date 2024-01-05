# Targets

Targets play a crucial role as the connection between observables measured in an 
experiment and the machine learning core behind BayBE.
In general, it is expected that you create one [`Target`](baybe.targets.base.Target) 
object for each of your observables. 
The way BayBE treats multiple targets is then controlled via the 
[`Objective`](../../userguide/objective).

