# Targets

## General information

For BaybBE, targets play a crucial role in modeling the function that requires optimization during the DoE campaign.
Targets are used to model the function that should be optimized by the DoE campaign. They are implemented via the [`Target`](baybe.targets.base.Target) class.
Each target has a name as well as an abstract transform function. This function is used to transform data into computational representation. 

BayBE currently supports a single type of target, called [`NumericalTarget`](baybe.targets.numerical.NumericalTarget), specifically designed for numerical modeling.

## Numerical targets

Besides their name, numerical targets have the following attributes.
* **The optimization mode:** This mode dictates whether we want to minimize or maximize the target or whether we want to match a specific value.
* **Bounds:** Define optional bounds that constrain the range of target values.
* **A bound transformtaion function:** When bounds are provided, a bound transformation function is used to map target values onto the [0,1] interval. This is only applicable in minimization and maximization mode.

Targets are used on all of the examples, mostly in either minimization or maximization mode. For an example on how the matching mode can be effectively utilized, we refer to the dedicated example provided [here](./../../examples/Multi_Target/desirability).