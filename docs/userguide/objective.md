# Objective

Optimization problems involve either a single target quantity of interest or 
several (potentially conflicting) targets that need to be considered simultaneously.  
BayBE uses the concept of an [`Objective`](baybe.objective.Objective) to allow the user
to control how these different types of scenarios are handled.

```{note}
We are actively working on adding more objective types for multiple targets.
```

## SingleTargetObjective
The need to optimize a single [`Target`](baybe.targets.base.Target) is the most basic
type of situation one can encounter in experimental design. 
In this scenario, the fact that only one target shall be considered in the design is
communicated to BayBE by wrapping the target into a
[`SingleTargetObjective`](baybe.objectives.single.SingleTargetObjective):
```python
from baybe.targets import NumericalTarget
from baybe.objectives import SingleTargetObjective

target = NumericalTarget(name="Yield", mode="MAX")
objective = SingleTargetObjective(target)
```
In fact, the role of the
`SingleTargetObjective` is to merely signal the absence of other `Target`s in the
optimization problem.
Because this fairly trivial conversion step requires no additional user configuration,
we provide a convenience constructor for it:

````{admonition} Convenience construction and implicit conversion
:class: tip
* The conversion from a single [`Target`](baybe.targets.base.Target) to a
[`SingleTargetObjective`](baybe.objectives.single.SingleTargetObjective) describes a
one-to-one relationship and can be triggered directly from the corresponding target
object:
  ```python
  objective = target.to_objective()
  ```
* Also, other class constructors that expect an 
[`Objective`](baybe.objectives.base.Objective)
object (such as [`Campaigns`](baybe.campaign.Campaign)) will happily accept
individual [`Targets`](baybe.targets.base.Target) instead and apply the necessary
conversion behind the scenes.
````

## DesirabilityObjective
The [`DesirabilityObjective`](baybe.objectives.desirability.DesirabilityObjective)
enables the combination of multiple targets via scalarization into a single numerical
value (commonly referred to as the *overall desirability*), a method also utilized in
classical DOE.

```{admonition} Mandatory target bounds
:class: attention
Since measurements of different targets can vary arbitrarily in scale, all targets
passed to a
[`DesirabilityObjective`](baybe.objectives.desirability.DesirabilityObjective) must be
normalizable to enable meaningful combination into desirability values. This requires
that all provided targets must have `bounds` specified (see [target user
guide](/userguide/targets.md)).
If provided, the necessary normalization is taken care of automatically. 
Otherwise, an error will be thrown.
```

Besides the list of `targets` to be scalarized, this objective type takes two
additional optional parameters that let us control its behavior:
* `weights`: Specifies the relative importance of the targets in the form of a
  sequence of positive numbers, one for each target considered.  
  **Note:** 
  BayBE automatically normalizes the weights, so only their relative
  scales matter.
* `scalarizer`: Specifies the [scalarization function](baybe.objectives.enum.Scalarizer)
  to be used for combining the normalized target values. 
  The choices are `MEAN` and `GEOM_MEAN`, referring to the arithmetic and 
  geometric mean, respectively.

The definitions of the `scalarizer`s are as follows, where $\{t_i\}$ enumerate the
**normalized** target measurements of single experiment and $\{w_i\}$ are the
corresponding target weights:

$$
\text{MEAN} &= \frac{1}{\sum w_i}\sum_{i} w_i \cdot t_i \\
\text{GEOM_MEAN} &= \left( \prod_i t_i^{w_i} \right)^{1/\sum w_i}
$$


In the example below, we consider three different targets (all associated with a
different goal) and give twice as much importance to the first target relative to each 
of the other two:
```python
from baybe.targets import NumericalTarget
from baybe.objectives import DesirabilityObjective

target_1 = NumericalTarget(name="t_1", mode="MIN", bounds=(0, 100))
target_2 = NumericalTarget(name="t_2", mode="MIN", bounds=(0, 100))
target_3 = NumericalTarget(name="t_3", mode="MATCH", bounds=(40, 60))
objective = DesirabilityObjective(
    targets=[target_1, target_2, target_3],
    weights=[2.0, 1.0, 1.0],  # optional (by default, all weights are equal)
    scalarizer="GEOM_MEAN",  # optional
)
```

For a complete example demonstrating desirability mode, see [here](./../../examples/Multi_Target/desirability).
