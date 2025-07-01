# Targets

Targets play a crucial role as the connection between observables measured in an
experiment and the machine learning core behind BayBE.
In general, it is expected that you create one [`Target`](baybe.targets.base.Target)
object for each of your observables.
The way BayBE treats multiple targets is then controlled via the 
[`Objective`](../../userguide/objectives).

## NumericalTarget
```{admonition} Important
:class: important

The {class}`~baybe.targets.numerical.NumericalTarget` class has been redesigned from the
ground up in version [0.14.0](https://github.com/emdgroup/baybe/releases/tag/0.13.1),
providing a more concise and significantly **more expressive interface.**

For a temporary transition period, the class constructor offers full backward
compatibility with the previous interface, meaning that it can be called with either the
new or the legacy arguments. However, this comes at the cost of **reduced typing
support**, meaning that you won't get type hints (e.g. for autocompletion or static type
checks) for either of the two types of calls. 

For this reason, we offer two additional constructors available for the duration of the
transition period that offer full typing support, which are useful for code development:
{meth}`~baybe.targets.numerical.NumericalTarget.from_legacy_interface` and
{meth}`~baybe.targets.numerical.NumericalTarget.from_modern_interface`.
```

Whenever you want to optimize a real-valued quantity, the 
{class}`~baybe.targets.numerical.NumericalTarget` class is the right choice.
Optimization with targets of this type follows two basic rules:
1. Targets are transformed as specified by their
   {class}`~baybe.transformations.base.Transformation` (with no transformation
   defined being equivalent to the identity transformation).
2. Whenever an optimization direction is required (i.e., when the context is *not*
   active learning), the transformed targets are assumed to be **maximized**.

This results in a simple yet flexible interface:
```python
from baybe.targets import NumericalTarget
from baybe.transformations import LogarithmicTransformation

target = NumericalTarget(
    name="Yield",
    transformation=LogarithmicTransformation(),  # optional transformation object
)
```

While the second rule may seem restrictive at first, it does not limit the
expressiveness of the resulting models, thanks to the transformation step applied.
In fact, other types of optimization problems (e.g., minimization, matching a
specific set point value, or pursuing any other custom objective) are just maximization
problems in disguise, hidden behind an appropriate target transformation.

For example:
* **Minimization** can be achieved by negating the targets before maximizing the
  resulting numerical values.
* **Matching** a set point value can be implemented by applying a transformation that
  computes the "proximity" to the set point in some way (e.g. in terms of the
  negative absolute difference to it).
* In general, any (potentially nonlinear) **custom objective** can be expressed using a
  transformation that assigns higher values to more desirable outcomes and lower values
  to less desirable outcomes.

Especially the first two cases are so common that we provide convenient ways to create
the corresponding target objects:

### Convenience Construction
Instead of manually providing the necessary transformation object, BayBE offers several
convenience approaches to construct targets for many common use cases.
The following is a non-comprehensive overview – for a complete list, please refer to the
[`NumericalTarget` documentation](baybe.targets.numerical.NumericalTarget).
* **Minimization**: Minimization of a target can be achieved by simply passing the
  `minimize=True` argument to the constructor:
  ```python
  target = NumericalTarget(
      name="Yield",
      transformation=LogarithmicTransformation(),  # optional transformation object
      minimize=True,  # this time, the yield is to be minimized
  )
  ```

  ````{admonition} Manual Inversion
  :class: note
  Note that the above is virtually the same as chaining the existing transformation with
  an inversion transformation:
  ```python
  from baybe.transformations import AffineTransformation
  target = NumericalTarget(
      name="Yield",
      transformation=LogarithmicTransformation() + AffineTransformation(factor=-1)
  )
  ```

  Implementation-wise, however, the two approaches differ in that the inversion is
  dynamically added before passing the target to the optimization algorithm in the
  former case, while it becomes an integral part of the target transformation attribute
  in the latter.
  ````

* **Matching a set point**: For common matching transformations, we provide
  convenience constructors with the `match_` prefix (see
  {class}`~baybe.targets.numerical.NumericalTarget` for all options).
  
  For example:
  ```python
  # Absolute transformation
  t_abs = NumericalTarget.match_absolute(name="Yield", match_value=42)  

  # Bell-shaped transformation
  t_bell = NumericalTarget.match_bell(name="Yield", match_value=42, sigma=5)

  # Triangular transformation
  t_tr1 = NumericalTarget.match_triangle(name="Yield", match_value=42, width=10)
  t_tr2 = NumericalTarget.match_triangle(name="Yield", match_value=42, cutoffs=(37, 47))
  t_tr3 = NumericalTarget.match_triangle(name="Yield", match_value=42, margins=(5, 5))
  assert t_tr1 == t_tr2 == t_tr3
  ```

* **Normalizing targets**: Sometimes, it is necessary to normalize the targets to a
  certain range, e.g. to ensure that values are always in the interval [0, 1]. 
  One situation where this can be required is when combining the targets using a
  {class}`baybe.objective.desirability.DesirabilityObjective`.
  For this purpose, we provide convenience constructors with the `normalize_` prefix
  (see {class}`~baybe.targets.numerical.NumericalTarget` for all options).
  
  For example:
  ```python
  target = NumericalTarget.normalize_ramp(name="Yield", bounds=(0, 1), descending=True)
  ```

* **Creation from existing targets**: Targets can also be quickly created from existing
  ones by calling certain transformation methods on them (see
  {class}`~baybe.targets.numerical.NumericalTarget` for all options).
  
  For example:
  ```python
  t1 = NumericalTarget("Yield")
  t2 = t1 - 10
  t3 = t2 * 5
  t4 = t3.abs()
  t5 = t4.power(2)
  t6 = t5.clamp(max=100)
  ```

## Limitations
```{important}
{class}`~baybe.targets.numerical.NumericalTarget` enables many use cases due to the
real-valued nature of most measurements. However, it can also be used to model
categorical targets if they are ordinal.

**For example:**
If your experimental outcome is a categorical ranking into "bad", "mediocre" and "good",
you could use a {class}`~baybe.targets.numerical.NumericalTarget` with bounds (1, 3)
and pre-map the categories to the values 1, 2 and 3, respectively.

If your target category is not ordinal, the transformation into a numerical target is
not straightforward, which is a current limitation of BayBE. We are looking into adding
more target options in the future.
```