# Environment Variables

Several aspects of BayBE can be configured via environment variables.

## Basic Instructions
Setting an environment variable with the name `ENVVAR_NAME` is best done before calling
any Python code, and must also be done in the same session unless made persistent, e.g.
via `.bashrc` or similar:
```bash
ENVAR_NAME="my_value"
python do_baybe_work.py
```
Or on Windows:
```shell
set ENVAR_NAME=my_value
```
Note that variables set in this manner are interpreted as text, but converted internally
to the needed format. See for instance the [`strtobool`](baybe.utils.boolean.strtobool) 
converter for values that can be set so BayBE can interpret them as Booleans.

It is also possible to set environment variables in Python:
```python
import os

os.environ["ENVAR_NAME"] = "my_value"

# proceed with BayBE code ...
```
However, this needs to be done carefully at the entry point of your script or session and
will not persist between sessions.


## Polars
If BayBE was installed with the additional `polars` dependency (`baybe[polars]`), it
will use the advanced methods of Polars to create the searchspace lazily and perform a
streamed evaluation of constraints. This will improve speed and memory consumption
during this process, and thus might be beneficial for very large search spaces.

Since this is still somewhat experimental, you might want to deactivate Polars without
changing the Python environment. To do so, you can set the environment variable 
`BAYBE_DEACTIVATE_POLARS` to any truthy value accepted by
[`strtobool`](baybe.utils.boolean.strtobool).

```{admonition} Row Order
:class: caution

For performance reasons, search space manipulation using `polars` is not
guaranteed to produce the same row order as the corresponding `pandas` operations.
```

## Disk Caching
For some components, such as the
[`SubstanceParameter`](baybe.parameters.substance.SubstanceParameter), some of the
computation results are cached in local storage.

By default, BayBE determines the location of temporary files on your system and puts
cached data into a subfolder `.baybe_cache` there. If you want to change the location of
the disk cache, change:
```bash
BAYBE_CACHE_DIR="/path/to/your/desired/cache/folder"
```

By setting
```bash
BAYBE_CACHE_DIR=""
```
you can turn off disk caching entirely.

## EXPERIMENTAL: Floating Point Precision
In general, double precision is recommended because numerical stability during optimization
can be bad when single precision is used. This impacts gradient-based optimization,
i.e. search spaces with continuous parameters, more than optimization without gradients.

If you still want to use single precision, you can set the following Boolean variables:
- `BAYBE_NUMPY_USE_SINGLE_PRECISION` (defaults to `False`)
- `BAYBE_TORCH_USE_SINGLE_PRECISION` (defaults to `False`)

```{admonition} Experimental Feature
:class: warning
Currently, it cannot be guaranteed that all calculations will be performed in single precision,
even when setting the aforementioned variables. The reason is that there are several code snippets
within `BoTorch` that transform single precision variables to double precision variables.
Consequently, this feature is currently only available as an *experimental* feature.
We are however actively working on fully enabling single precision.
```

## Parallel Runs in Scenario Simulations
By default, [`simulate_scenarios`](baybe.simulation.scenarios.simulate_scenarios)
function is to run in parallel. This can be disabled by setting the environment variable `BAYBE_PARALLEL_SIMULATION_RUNS` to a [falsy value](baybe.utils.boolean.strtobool):

```bash
BAYBE_PARALLEL_SIMULATION_RUNS="False"  # Set this to disable parallel execution
```

Alternatively, you can directly specify the `parallel_runs` parameter when calling the function, which takes precedence over the environment variable:

~~~python
from baybe.simulation import simulate_scenarios

results = simulate_scenarios(
    scenarios=scenarios,
    lookup=lookup,
    n_mc_iterations=10,
    parallel_runs=False,  # Disable parallel runners for this call
)
~~~

```{admonition} Experimental Feature
:class: warning
While parallel execution usually speeds up computation significantly, the performance
impact can vary depending on the machine and simulation configuration. In some cases, it
might even lead to longer run times due to overhead costs.
```

## FPS Sampling Implementation
The optional package `fpsample` can be installed, e.g. by specifying the optional
dependency group in `pip install baybe[extras]`. This prompts BayBE to use an 
implementation of [farthest point sampling](baybe.recommenders.pure.nonpredictive.sampling.FPSRecommender)
which is far more memory-efficient and scalable. This implementation, however, requires
setting [`FPSRecommender.initialization="farthest"`](baybe.recommenders.pure.nonpredictive.sampling.FPSRecommender.initialization)
and [`FPSRecommender.random_tie_break=False`](baybe.recommenders.pure.nonpredictive.sampling.FPSRecommender.random_tie_break).

In case you want to use other settings while having `fpsample` installed, you can
bypass the use of `fpsample` via an environment variable:

```bash
BAYBE_USE_FPSAMPLE="False"  # Do not use `fpsample` even if installed
```