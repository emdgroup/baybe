# Objective

The [`Objective`](baybe.objective.Objective) instructs BayBE how to deal with multiple 
targets (if applicable).

To create an objective, it is necessary to provide at the following:
* An optimization ``mode``: This can be either ``SINGLE`` or ``DESIRABILITY``, 
  denoting either the optimization of a single target function or a combination of 
  different target functions.
* A list of ``targets``: The list of targets that are optimized. Note that the 
  ``SINGLE`` mode also requires a list containing the single target.

```{note}
We are actively working on adding more objective modes for multiple targets.
```

## Supported Optimization Modes
Currently, BayBE offers two optimization modes.

### ``SINGLE``
In the ``SINGLE`` mode, objectives focus on optimizing a single target. 
Nearly all of the [examples](../../examples/examples) use this objective mode.

```python
from baybe.targets import NumericalTarget
from baybe.objective import Objective

target1 = NumericalTarget(name="yield", mode="MIN", bounds=(0, 100))
objective = Objective(mode="SINGLE", targets=[target1])
```
