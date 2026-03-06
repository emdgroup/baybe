# Objective

Optimization problems involve either a single target quantity of interest or several
(potentially conflicting) targets that need to be considered simultaneously. BayBE uses
the concept of an [`Objective`](baybe.objectives.base.Objective) to allow the user to
control how these different types of scenarios are handled.

## Objective Types
The following objective types are available:

### SingleTargetObjective
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

### DesirabilityObjective
The [`DesirabilityObjective`](baybe.objectives.desirability.DesirabilityObjective)
enables the combination of multiple targets via scalarization into a single numerical
value (commonly referred to as the *overall desirability*), a method also utilized in
classical DOE.

(target-normalization)=
```{admonition} Target Normalization
:class: important
Since desirability computation relies on scalarization, and because targets can vary
arbitrarily in scale, it is (by default) required that all targets are properly
normalized before entering the computation to enable meaningful combination into
desirability values. This can be achieved by applying appropriate normalizing target
[transformations](/userguide/transformations).

Alternatively, if you know what you are doing, you can also disable this requirement
via the [`require_normalization`](#require-normalization) flag. 
```

Besides the list of [`Target`](baybe.targets.base.Target)s to be scalarized, this
objective type takes additional optional arguments that let us control its behavior:

* {attr}`~baybe.objectives.desirability.DesirabilityObjective.weights`:
  Specifies the relative importance of the targets in the form of a sequence of positive
  numbers, one for each target considered. Note that BayBE automatically normalizes the
  weights, so only their relative scales matter.

* {attr}`~baybe.objectives.desirability.DesirabilityObjective.as_pre_transformation`:
  By default, the desirability is computed via posterior transformations, enabling
  access to information even for the original targets (e.g. in
  [`SHAPInsight`](baybe.insights.shap.SHAPInsight) or
  [`posterior_stats`](baybe.campaign.Campaign.posterior_stats)). However, this requires
  one model per target. With
  [`as_pre_transformation=True`](baybe.objectives.desirability.DesirabilityObjective.as_pre_transformation)
  you can change this behavior, e.g. due to computational limitations. The objective
  will then apply the transformation directly and fits a single model on the resulting
  "Desirability".
 
  (require-normalization)=
* {attr}`~baybe.objectives.desirability.DesirabilityObjective.require_normalization`:
  A Boolean flag controlling the target normalization requirement.

* {attr}`~baybe.objectives.desirability.DesirabilityObjective.scalarizer`:
  Specifies the [scalarization function](baybe.objectives.enum.Scalarizer) to be used
  for combining the normalized target values.


The definitions of the
{attr}`~baybe.objectives.desirability.DesirabilityObjective.scalarizer`s are as follows,
where $\{t_i\}$ refer the **transformed** target measurements of a single experiment and
$\{w_i\}$ are the corresponding target weights:

```{math}
\begin{align*}
  \text{MEAN} &= \frac{\sum_{i} w_i \cdot t_i}{\sum_{i} w_i} \\
  \text{GEOM_MEAN} &= \left( \prod_i t_i^{w_i} \right)^{1/\sum_{i} w_i}
\end{align*}
```

#### Example 1 – Normalized Targets
Here, we consider four **normalized** targets, each with a distict
{ref}`optimization goal <userguide/targets:NumericalTarget>` chosen arbitrarily
for demonstration purposes. The first target is given twice as much importance as each
of the other three by assigning it a higher weight:
```python
from baybe.targets import NumericalTarget
from baybe.objectives import DesirabilityObjective

t1 = NumericalTarget.normalized_ramp(name="t1", cutoffs=(0, 100), descending=True)
t2 = NumericalTarget.normalized_sigmoid(name="t2", anchors=[(0, 0.1), (100, 0.9)])
t3 = NumericalTarget.match_bell(name="t3", match_value=50, sigma=10)
t4 = NumericalTarget(name="t4").exp().clamp(max=10).normalize()
objective = DesirabilityObjective(
    targets=[t1, t2, t3, t4],
    weights=[2.0, 1.0, 1.0, 1.0],  # optional (by default, all weights are equal)
    scalarizer="GEOM_MEAN",  # optional
)
```

#### Example 2 – Non-Normalized Targets
Sometimes, we explicitly want to bypass the normalization requirement, for example,
when the target ranges are unknown. In this case, using the unweighted arithmetic mean
is a reasonable choice:
```python
from baybe.targets import NumericalTarget
from baybe.objectives import DesirabilityObjective

t1 = NumericalTarget(name="t_max")
t2 = NumericalTarget.match_absolute(name="t_match", match_value=0)
objective = DesirabilityObjective(
    targets=[t1, t2],
    scalarizer="MEAN",
    require_normalization=False,  # disable normalization requirement
)
```

### ParetoObjective
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
the flexibility to adjust their preferences post-hoc – knowing that each of the obtained
points is optimal with respect to a particular preference model.

To set up a [`ParetoObjective`](baybe.objectives.pareto.ParetoObjective), simply
specify the corresponding target objects:
```python
from baybe.targets import NumericalTarget
from baybe.objectives import ParetoObjective

target_1 = NumericalTarget(name="t_1")
target_2 = NumericalTarget(name="t_2", minimize=True)
target_3 = NumericalTarget.match_absolute(name="t_3", match_value=0)
objective = ParetoObjective(targets=[target_1, target_2, target_3])
```

```{admonition} Convenience Multi-Output Casting
:class: tip
[`ParetoObjective`](baybe.objectives.pareto.ParetoObjective) requires a 
[multi-output surrogate model](multi_output_modeling). 
If you attempt to use a  single-output model, BayBE will automatically turn it into a 
[`CompositeSurrogate`](baybe.surrogates.composite.CompositeSurrogate) 
using [independent replicates](auto_replication).
```

## Identifying Non-Dominated Configurations
The {meth}`~baybe.objectives.base.Objective.identify_non_dominated_configurations`
method provides a straightforward way to identify *non-dominated* target configurations
(i.e., those lying on the Pareto front) for any chosen
{class}`~baybe.objectives.base.Objective`. The result is a Boolean
{class}`~pandas.Series` highlighting the corresponding subset among the passed set of
configurations.

```{admonition} Domination for Non-Pareto Objectives
:class: note

While the concept of (non-)domination is typically associated with Pareto optimization,
it can also be applied to objectives of types other than
{class}`~baybe.objectives.pareto.ParetoObjective`. In general, we identify non-dominated
configurations as the subset of configurations that are Pareto-optimal as measured by
their **transformed representations** induced by the transformation rules of the
used objective and involved targets. In other words, the *regular* Pareto criterion is
applied on the *transformed* space of computed objective values.

This implies two noteworthy special cases:
* For {class}`~baybe.objectives.pareto.ParetoObjective`, it recovers the standard logic
where transformation axes are defined by individual (transformed) targets.
* For {class}`~baybe.objectives.single.SingleTargetObjective`, it corresponds to the
regular optimization of the respective target, where the Pareto front collapses to a
single point representing the optimal target value.
```

```python
import pandas as pd

from baybe.objectives import ParetoObjective
from baybe.targets import NumericalTarget

objective = ParetoObjective(
    targets=[
        NumericalTarget("t_max"),
        NumericalTarget("t_min", minimize=True),
    ]
)
configurations = pd.DataFrame(
    {
        "t_max": [0.1, 0.3, 0.5, 0.9],
        "t_min": [0.9, 0.2, 0.6, 0.5],
    }
)
mask = objective.identify_non_dominated_configurations(configurations)
assert mask.equals(pd.Series([False, True, False, True]))
```

