# Targets

Targets play a crucial role as the connection between observables measured in an
experiment and the machine learning core behind BayBE.
In general, it is expected that you create one [`Target`](baybe.targets.base.Target)
object for each of your observables to inform BayBE about their existence.
The way BayBE treats these targets is then controlled via the
[`Objective`](../../userguide/objectives).

## NumericalTarget
```{admonition} Important
:class: important

The {class}`~baybe.targets.numerical.NumericalTarget` class has been redesigned from the
ground up in version [0.14.0](https://github.com/emdgroup/baybe/releases/),
providing a more concise and significantly **more expressive interface.**

For a temporary transition period, the class constructor offers full backward
compatibility with the previous interface, meaning that it can be called with either the
new or the legacy arguments. However, this comes at the cost of **reduced typing
support**, meaning that you won't get type hints (e.g. for autocompletion or static type
checks) for either of the two types of constructor calls.

For this reason, we offer two additional constructors available **for the duration of
the deprecation period** that offer full typing support:
{meth}`~baybe.targets.numerical.NumericalTarget.from_legacy_interface` and
{meth}`~baybe.targets.numerical.NumericalTarget.from_modern_interface`.
```

Use the {class}`~baybe.targets.numerical.NumericalTarget` class for optimizing
**real-valued** quantities.
Optimization with targets of this type follows two basic rules:
1. Targets are transformed as specified by their
   {attr}`~baybe.targets.numerical.NumericalTarget.transformation`, with no
   transformation defined being equivalent to the identity transformation.
2. Whenever an optimization direction is required (i.e., when the context is *not*
   [active learning](/userguide/active_learning)), the transformed targets are assumed
   to be **maximized** by default or **minimized** if explicitly specified via their
   {attr}`~baybe.targets.numerical.NumericalTarget.minimize` flag.

This results in a simple yet expressive interface:
```python
from baybe.targets import NumericalTarget
from baybe.transformations import LogarithmicTransformation

target = NumericalTarget(
    name="Yield",
    transformation=LogarithmicTransformation(),  # optional transformation
    minimize=False,  # this is the default
)
```

(targets-as-instruction)=
```{admonition} Targets are Optimization Instructions
:class: note

Notice how the target ingredients above declaratively specify the **different aspects**
of the underlying optimization problem:
* The {attr}`~baybe.targets.numerical.NumericalTarget.name` defines the signal
  **"source"**, i.e. the observable being measured.
* The {attr}`~baybe.targets.numerical.NumericalTarget.transformation` defines the
  **"what"**, i.e. which derivative of the signal is to be optimized.
* The {attr}`~baybe.targets.numerical.NumericalTarget.minimize` flag defines the
  **"how"**, i.e. the desired optimization direction.
```

While the second rule may seem restrictive at first, it does not limit the
expressiveness of the resulting models, thanks to the transformation step applied. In
fact, all types of optimization problems (e.g., minimization, matching/avoiding one or
[multiple set point values](../../examples/Transformations/laser_tuning), or pursuing
any other custom objective) are just maximization problems in disguise, hidden behind an
appropriate target transformation.


For example:
* **Minimization** can be achieved by negating the targets before maximizing the
  resulting numerical values. For more information, see [here](#minimization).
* **Matching** a set point value can be implemented by applying a transformation that
  computes the "proximity" to the set point in some way (e.g. in terms of the
  negative absolute difference to it). Similarly, avoiding the set point can be
  achieved by reversing the sign of the proximity measure (or activating the
  {attr}`~baybe.targets.numerical.NumericalTarget.minimize` flag in addition).
  For more information, see [here](#set-point-matching).
* In general, any (potentially nonlinear) **custom objective** can be expressed using a
  transformation that assigns higher values to more desirable outcomes and lower values
  to less desirable outcomes. Examples can be found 
  [here](../../examples/Transformations/Transformations).

Many cases – especially the first two described above – are so common that we offer
convenient ways to directly create the corresponding target objects for many
optimization workflows, eliminating the need to manually specify the necessary
{class}`~baybe.transformations.base.Transformation` object yourself:

### Minimization

Minimization of a target can be achieved by simply passing the `minimize=True` argument
to the constructor:
```python
from baybe.targets import NumericalTarget

t = NumericalTarget(
    name="Cost",
    minimize=True,  # cost is to be minimized
)
```

````{admonition} Minimization = Negated Maximization
:class: caution
Behind the scenes, minimization of targets is achieved by maximizing their negated
values: the {attr}`~baybe.targets.numerical.NumericalTarget.minimize` flag is used to
inform the corresponding {class}`~baybe.objectives.base.Objective` holding the
{class}`~baybe.targets.numerical.NumericalTarget` object to inject an appropriate
negating transformation **just before** passing the target values to the optimization
engine, allowing us to reuse the same maximization-based routines for all targets. The
details of this negation step depend on the objective type being used.

However, while numerically equivalent, there is a semantic difference between minimizing a
quantity and maximizing the negated signal derived from it. This difference is both
reflected by [the way targets are specified](#targets-as-instruction) as
well as by the resulting objects:
```python
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from baybe.targets import NumericalTarget
from baybe.transformations import AffineTransformation

# Target 1: "Minimize" cost
t1 = NumericalTarget(name="Cost", minimize=True)

# Target 2: "Maximize" the quantity obtained from negating cost measurements
t2 = NumericalTarget(name="Cost", transformation=AffineTransformation(factor=-1))

# Although both targets yield the same objective values ...
s = pd.Series(np.linspace(0, 10), name="Cost")
df = s.to_frame()
assert_frame_equal(
    t1.to_objective().transform(df),
    t2.to_objective().transform(df),
)

# ... the targets themselves are not equal ...
assert t1 != t2

# ... and the transformed signals they specify differ!
assert not t1.transform(s).equals(t2.transform(s))
```
````

### Set Point Matching
For common matching transformations, we provide convenience constructors with the
`match_` prefix (see {class}`~baybe.targets.numerical.NumericalTarget` for all options).
Similar to [minimization](#minimization) targets, these constructors inject a
suitable transformation computing some form of "proximity" to the set point value.
  
While you can easily implement your own (potentially complex) matching logic using the
{class}`~baybe.transformations.basic.CustomTransformation` class, let us have a look at
how we can match a single set point using built-in constructors:


#### Absolute Transformation

![Transforms](../_static/targets/absolute.svg)

The potentially simplest way to match a set point value is by minimizing the absolute
distance to it, since it requires no configuration other than specifying the set point
value itself. The {meth}`~baybe.targets.numerical.NumericalTarget.match_absolute`
constructor allows you to do exactly that in a single line of code.

**Example**
```python
t_abs = NumericalTarget.match_absolute(name="Size", match_value=42)
```

```{admonition} Practical Considerations
:class: note
✅ Simple to use: no configuration required other than the set point value itself

❌ Cannot be used in situations where normalized targets are required
```

#### Triangular Transformation

![Transforms](../_static/targets/triangular.svg)

In some cases, we want to penalize [absolute distance](#absolute-transformation) to the set point but only
up to a certain threshold, above which any further deviation does not matter. For this purpose,
the {meth}`~baybe.targets.numerical.NumericalTarget.match_triangular` constructor can be used,
which allows us to specify these thresholds in various ways.

**Example**
```python
t1 = NumericalTarget.match_triangular(name="Size", match_value=42, width=10)
t2 = NumericalTarget.match_triangular(name="Size", match_value=42, cutoffs=(37, 47))
t3 = NumericalTarget.match_triangular(name="Size", match_value=42, margins=(5, 5))
assert t1 == t2 == t3
```

```{admonition} Practical Considerations
:class: note
✅ Normalized output: enables direct comparison with other normalized targets

✅ Possibility to directly specify an "acceptable range" around the set point value 

❌ Outside the triangular region, the gradient is zero, which can complicate
optimization if the thresholds are chosen too tight
```

#### Bell Transformation

![Transforms](../_static/targets/bell.svg)

Bell-transformed targets created via the
{meth}`~baybe.targets.numerical.NumericalTarget.match_bell` constructor can be
considered relaxed versions of their [triangular](#triangular-transformation)
counterparts. Unlike the latter, they have no strict cutoff points, resulting in a smooth
change in the output with non-zero gradient on the entire domain.

**Example**
```python
t_bell = NumericalTarget.match_bell(name="Size", match_value=42, sigma=5)
```

```{admonition} Practical Considerations
:class: note
✅ Normalized output: enables direct comparison with other normalized targets

✅ Smooth gradient on the entire domain, which can be beneficial for optimization

❌ Width of the bell is sometimes not intuitive to set
```

#### Power Transformation

![Transforms](../_static/targets/power.svg)

If you need more precise control over how strongly deviations from the set point are
penalized, you can use the {meth}`~baybe.targets.numerical.NumericalTarget.match_power`
constructor, which applies a power transformation to the [absolute distance](#absolute-transformation).
For the common case of squared penalties, we also provide a separate
{meth}`~baybe.targets.numerical.NumericalTarget.match_quadratic` constructor.

**Example**
```python
t_power = NumericalTarget.match_power(name="Size", match_value=42, exponent=2)
t_quad = NumericalTarget.match_quadratic(name="Size", match_value=42)
assert t_power == t_quad
```

```{admonition} Practical Considerations
:class: note
✅ Offers control over how strongly deviations from the set point are penalized

✅ Smooth gradient on the entire domain, which can be beneficial for optimization

❌ Cannot be used in situations where normalized targets are required
```

#### Custom Transformation
If none of the built-in constructors fit your needs because you need more fine-grained
control over the matching behavior (e.g. when there are multiple acceptable set points),
you always have the fallback option to create a
{class}`~baybe.transformations.basic.CustomTransformation` that implements the
corresponding logic and pass it to the regular
{class}`~baybe.targets.numerical.NumericalTarget` constructor.


### Target Normalization
Sometimes, it is necessary to normalize targets to the interval [0, 1] – especially when
multiple targets are present – in order to align them on a common scale. One situation
where this can be required is when combining the targets using a
{class}`~baybe.objectives.desirability.DesirabilityObjective`. For this purpose, we
provide convenience constructors with the `normalized_` prefix:

#### Ramp Transformation

![Transforms](../_static/targets/ramp.svg)

The {meth}`~baybe.targets.numerical.NumericalTarget.normalized_ramp` constructor offers
the simplest way to create a normalized target. It does so by linearly mapping the
target values to the range [0, 1] inside a specified interval and clamping the output
outside.

**Example**
```python
t = NumericalTarget.normalized_ramp(name="Target", cutoffs=(0, 1), descending=True)
```

```{admonition} Practical Considerations
:class: note
✅ Easy to interpret: output value changes linearly inside the specified range

❌ Outside the linear region, the gradient is zero, which can complicate
optimization if the thresholds are chosen too tight
```

#### Sigmoid Transformation

![Transforms](../_static/targets/sigmoid.svg)

The {meth}`~baybe.targets.numerical.NumericalTarget.normalized_sigmoid` constructor
can be considered a softened version of the [ramp transformation](#ramp-transformation).
Instead of using hard cutoffs, it smoothly interpolates the target values between 
0 and 1 using a sigmoid function. 

**Example**
```python
t = NumericalTarget.normalized_sigmoid(name="Target", anchors=[(-1, 0.1), (1, 0.9)])
```

```{admonition} Practical Considerations
:class: note
✅ Smooth gradient on the entire domain, which can be beneficial for optimization

❌ Requires more parameters to configure than the [ramp transformation](#ramp-transformation)
```


#### Normalizing Existing Targets
You can also create a normalized version of an existing target by calling its
{meth}`~baybe.targets.numerical.NumericalTarget.normalize` method, provided the target
already maps to a bounded domain. For brevity and demonstration purposes, we show an
example using [method chaining](method-chaining): 

```python
t = NumericalTarget(name="Target").power(2).clamp(max=1).normalize()
```

(method-chaining)=
### Creation From Existing Targets
Targets can also be quickly created from existing ones by calling certain transformation
methods on them (see {class}`~baybe.targets.numerical.NumericalTarget` for all options).

For example:
```python
t1 = NumericalTarget("Target")
t2 = t1 - 1  # subtract a constant
t3 = t2 / 5  # divide by a constant
t4 = t3.abs()  # compute absolute value
t5 = t4.power(3)  # compute the cube
t6 = t5.clamp(max=10)  # upper-bound to 10 (lower bound is 0 due to abs() call above)
t7 = t6.normalize()  # normalize to [0, 1]
```

## Limitations
```{important}
{class}`~baybe.targets.numerical.NumericalTarget` enables many use cases due to the
real-valued nature of most measurements. However, it can also be used to model
categorical targets if they are ordinal.

**For example:**
If your experimental outcome is a categorical ranking into "bad", "mediocre" and "good",
you could use a {class}`~baybe.targets.numerical.NumericalTarget`
by pre-mapping the categories to the values 1, 2 and 3, respectively.

If your target category is not ordinal, the transformation into a numerical target is
not straightforward, which is a current limitation of BayBE. We are looking into adding
more target variants in the future.
```