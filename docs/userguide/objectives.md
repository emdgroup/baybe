# Objective

Optimization problems involve either a single target quantity of interest or several
(potentially conflicting) targets that need to be considered simultaneously. BayBE uses
the concept of an [`Objective`](baybe.objectives.base.Objective) to allow the user to
control how these different types of scenarios are handled.

## SingleTargetObjective
The need to optimize a single [`Target`](baybe.targets.base.Target) is the most basic
type of situation one can encounter in experimental design. 
In this scenario, the fact that only one target shall be considered in the design is
communicated to BayBE by wrapping the target into a
[`SingleTargetObjective`](baybe.objectives.single.SingleTargetObjective):
```python
from baybe.targets import NumericalTarget
from baybe.objectives import SingleTargetObjective

target = NumericalTarget(name="Yield")
objective = SingleTargetObjective(target)
```
In fact, the role of the
[`SingleTargetObjective`](baybe.objectives.single.SingleTargetObjective) 
is to merely signal the absence of other [`Targets`](baybe.targets.base.Target)
in the optimization problem.
Because this fairly trivial conversion step requires no additional user configuration,
we provide a convenience constructor for it:

````{admonition} Convenience Construction and Implicit Conversion
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

```{admonition} Mandatory Target Bounds
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

Besides the list of [`Targets`](baybe.targets.base.Target)
to be scalarized, this objective type takes two
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

target_1 = NumericalTarget.normalize_ramp(name="t_1", bounds=(0, 100))
target_2 = NumericalTarget.normalize_ramp(name="t_2", bounds=(0, 100), descending=True)
target_3 = NumericalTarget.match_bell(name="t_3", match_value=50, sigma=10)
objective = DesirabilityObjective(
    targets=[target_1, target_2, target_3],
    weights=[2.0, 1.0, 1.0],  # optional (by default, all weights are equal)
    scalarizer="GEOM_MEAN",  # optional
)
```

For a complete example demonstrating desirability mode, see [here](./../../examples/Multi_Target/desirability).

## ParetoObjective
The [`ParetoObjective`](baybe.objectives.pareto.ParetoObjective) can be used when the
goal is to find a set of solutions that represent optimal trade-offs among
multiple conflicting targets. Unlike the
[`DesirabilityObjective`](#DesirabilityObjective), this approach does not aggregate the
targets into a single scalar value but instead seeks to identify the Pareto front – the
set of *non-dominated* target configurations.

```{admonition} Non-Dominated Configurations
:class: tip
A target configuration is considered non-dominated (or Pareto-optimal) if no other
configuration is better in *all* targets.
```

Identifying the Pareto front requires maintaining explicit models for each of the
targets involved. Accordingly, it requires to use acquisition functions capable of
processing vector-valued input, such as
{class}`~baybe.acquisition.acqfs.qLogNoisyExpectedHypervolumeImprovement`. This differs
from the [`DesirabilityObjective`](#DesirabilityObjective), which relies on a single
predictive model to describe the associated desirability values. However, the drawback
of the latter is that the exact trade-off between the targets must be specified *in
advance*, through explicit target weights. By contrast, the Pareto approach allows to
specify this trade-off *after* the experiments have been carried out, giving the user
the flexibly to adjust their preferences post-hoc – knowing that each of the obtained
points is optimal with respect to a particular preference model.

To set up a [`ParetoObjective`](baybe.objectives.pareto.ParetoObjective), simply
specify the corresponding target objects:
```python
from baybe.targets import NumericalTarget
from baybe.objectives import ParetoObjective

target_1 = NumericalTarget(name="t_1")
target_2 = NumericalTarget(name="t_2", minimize=True)
target_3 = NumericalTarget.match_absolute(name="t_3", match_value=0)
objective = ParetoObjective(targets=[target_1, target_2])
```

```{admonition} Convenience Multi-Output Casting
:class: tip
[`ParetoObjective`](baybe.objectives.pareto.ParetoObjective) requires a 
[multi-output surrogate model](multi_output_modeling). 
If you attempt to use a  single-output model, BayBE will automatically turn it into a 
[`CompositeSurrogate`](baybe.surrogates.composite.CompositeSurrogate) 
using [independent replicates](auto_replication).
```
