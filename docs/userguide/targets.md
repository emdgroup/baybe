# Targets

Targets play a crucial role as the connection between observables measured in an
experiment and the machine learning core behind BayBE.
In general, it is expected that you create one [`Target`](baybe.targets.base.Target)
object for each of your observables.
The way BayBE treats multiple targets is then controlled via the 
[`Objective`](../../userguide/objectives).

## NumericalTarget
Besides the `name`, a [`NumericalTarget`](baybe.targets.numerical.NumericalTarget)
has the following attributes:
* **The optimization** `mode`: Specifies whether we want to minimize/maximize
  the target or whether we want to match a specific value.
* **Bounds**: Defines `bounds` that constrain the range of target values.
* **A** `transformation` **function**: When bounds are provided, this is
  used to map target values into the [0, 1] interval.

Below is a visualization of possible choices for `transformation`, where `lower` and
`upper` are the entries provided via `bounds`:
![Transforms](../_static/target_transforms.svg)

### MIN and MAX mode
Here are two examples for simple maximization and minimization targets:
```python
from baybe.targets import NumericalTarget, TargetMode, TargetTransformation

max_target = NumericalTarget(
    name="Target_1",
    mode=TargetMode.MAX,  # can also be provided as string "MAX"
)

min_target = NumericalTarget(
    name="Target_2",
    mode="MIN",  # can also be provided as TargetMode.MIN
    bounds=(0, 100),  # optional
    transformation=TargetTransformation.LINEAR,  # optional, will be applied if bounds are not None
)
```

### MATCH mode
If you want to match a desired value, the `TargetMode.MATCH` mode is the right choice.
In this mode, `bounds` are required and different transformations compared to `MIN`
and `MAX` modes are allowed.

Assume we want to instruct BayBE to match a value of 50 in a target.
We simply need to choose the bounds so that the midpoint is the desired value.
The spread of the bounds interval defines how fast the acceptability of a measurement
falls off away from the match value, also depending on the choice of `transformation`.

In the example below, `match_targetA` will treat all values < 45 and > 55 as
equally bad, while `match_targetB` is more forgiving in that it chooses a bell curve
transformation instead of a triangular one, and also uses a wider interval of bounds.
Both targets are configured such that the midpoint of `bounds` (in this case 50) 
becomes the optimal value:

```python
from baybe.targets import NumericalTarget, TargetMode, TargetTransformation

match_targetA = NumericalTarget(
    name="Target_3A",
    mode=TargetMode.MATCH,
    bounds=(45, 55),  # mandatory in MATCH mode
    transformation=TargetTransformation.TRIANGULAR,  # optional, applied if bounds are not None
)
match_targetB = NumericalTarget(
    name="Target_3B",
    mode="MATCH",
    bounds=(0, 100),  # mandatory in MATCH mode
    transformation="BELL",  # can also be provided as TargetTransformation.BELL
)
```

Targets are used in nearly all [examples](../../examples/examples).

## Limitations
```{important}
At the moment, BayBE's only option for targets is the `NumericalTarget`.
This enables many use cases due to the real-valued nature of most measurements.
But it can also be used to model categorial targets if they are ordinal.
For example: If your experimental outcome is a categorical ranking into "bad",
"mediocre" and "good", you could use a NumericalTarget with bounds (1, 3), where the
categories correspond to values 1, 2 and 3 respectively.
If your target category is not ordinal, the transformation into a numerical target is
not straightforward, which is a current limitation of BayBE.
We are looking into adding more target options in the future.
```