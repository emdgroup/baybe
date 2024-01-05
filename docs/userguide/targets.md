# Targets

Targets play a crucial role as the connection between observables measured in an 
experiment and the machine learning core behind BayBE.
In general, it is expected that you create one [`Target`](baybe.targets.base.Target) 
object for each of your observables. 
The way BayBE treats multiple targets is then controlled via the 
[`Objective`](../../userguide/objective).

## ``NumericalTarget``
Besides their ``name``, numerical targets have the following attributes:
* **The optimization** ``mode``: This specifies whether we want to minimize or maximize 
  the target or whether we want to match a specific value.
* **Bounds:** Define ``bounds`` that constrain the range of target values.
* **A transformation function:** When bounds are provided, a ``transformation`` is 
  used to map target values onto the [0,1] interval.

