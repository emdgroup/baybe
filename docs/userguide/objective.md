# Objective

The [`Objective`](baybe.objective.Objective) instructs BayBE how to deal with multiple
targets (if applicable).

To create an objective, it is necessary to provide the following:
* An optimization `mode`: This can be either `SINGLE` or `DESIRABILITY`,
  denoting the optimization of a single target function or a combination of
  different target functions respectively.
* A list of `targets`: The list of targets that are optimized (see user guide for
  [`Target`](../../userguide/targets)). Note that the `SINGLE` mode also requires a
  list containing the single target.

```{note}
We are actively working on adding more objective modes for multiple targets.
```

## Supported Optimization Modes
Currently, BayBE offers two optimization modes.

### SINGLE
In `SINGLE` mode, objectives focus on optimizing a single target. 
Nearly [examples](../../examples/examples) use this objective mode.

```python
from baybe.targets import NumericalTarget
from baybe.objective import Objective

target_1 = NumericalTarget(name="yield", mode="MIN", bounds=(0, 100))
objective = Objective(mode="SINGLE", targets=[target_1])
```

### DESIRABILITY
The `DESIRABILITY` mode enables the combination multiple targets via scalarization 
into a single value, a method also utilized in classical DOE.

Besides `mode` and `targets`, this objective type takes two additional optional
arguments:
* `weights`: Some targets might be more important than others.
  It is possible to specify the relative weights of the targets in this argument.
  BayBE automatically normalizes the numbers provided, so only their relative values 
  matter.
* `combine_func`: Specifies the function used for combining the transformed targets. 
  The choices are `MEAN` and `GEOM_MEAN`, referring to the arithmetic and 
  geometric mean respectively.

The definitions of the means are as follows, where $\{t_i\}$ enumerate the **scaled**
target observations for a single measurement and $\{w_i\}$ are the weights associated
with the respective target:

$$
\text{MEAN} &= \frac{1}{\sum w_i}\sum_{i} w_i \cdot t_i \\
\text{GEOM_MEAN} &= \left( \prod_i t_i^{w_i} \right)^{1/\sum w_i}
$$

```{admonition} Mandatory Target Bounds
:class: attention
Due to the combination of targets of potentially different scale, in `DESIRABILITY` 
objective mode, all provided targets must have `bounds` specified so they can be 
normalized via scaling before being combined.
```

In the example below, we use three different targets (which all have a different goal) 
and weigh the first target twice as important as each of the other targets:
```python
from baybe.targets import NumericalTarget
from baybe.objective import Objective

target_1 = NumericalTarget(name="t_1", mode="MIN", bounds=(0, 100))
target_2 = NumericalTarget(name="t_2", mode="MIN", bounds=(0, 100))
target_3 = NumericalTarget(name="t_3", mode="MATCH", bounds=(40, 60))
objective = Objective(
    mode="DESIRABILITY",
    targets=[target_1, target_2, target_3],
    weights=[2.0, 1.0, 1.0],  # optional, by default all weights are equal
    combine_func="GEOM_MEAN",  # optional, geometric mean is the default
)
```

For a complete example demonstrating desirability mode, see [here](./../../examples/Multi_Target/desirability).