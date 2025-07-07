# Transformations

Transformations allow you to customize the way numerical quantities enter the
recommendation process. They can be used to express various optimization objectives and
imprint your domain knowledge or use case requirements on the quantities being
optimized.

```{admonition} Note
:class: note
Currently, transformations are only used for the
{class}`~baybe.targets.numerical.NumericalTarget` class but it is planned to enable
their use for {class}`~baybe.parameters.numerical.NumericalContinuousParameter` as
well.
```

## Pre-defined Transformations

The following pre-defined transformation types are available via the
{mod}`baybe.transformations` module: 

### IdentityTransformation

The {class}`~baybe.transformations.core.IdentityTransformation` is a no-op transformation
that leaves the input values unchanged.

`````{grid} 2

````{grid-item}
:columns: auto

![Transforms](../_static/favicon.ico)
````

````{grid-item}
:columns: auto

**Transformation rule**

```{math}
f(x) = x
```
````
`````

**Example**

```python
from baybe.transformations import IdentityTransformation

t = IdentityTransformation()  # nothing to be configured here
```

### AbsoluteTransformation

The {class}`~baybe.transformations.core.AbsoluteTransformation` computes
the absolute value of the given input.

`````{grid} 2

````{grid-item}
:columns: auto
![Transforms](../_static/favicon.ico)
````

````{grid-item}
:columns: auto

**Transformation rule**
```{math}
f(x) = |x|
```
````
`````

**Example**

```python
from baybe.transformations import AbsoluteTransformation

t = AbsoluteTransformation()  # nothing to be configured here
```

### ExponentialTransformation

The {class}`~baybe.transformations.core.ExponentialTransformation` computes the
exponential of the given input.

`````{grid} 2

````{grid-item}
:columns: auto

![Transforms](../_static/favicon.ico)
````

````{grid-item}
:columns: auto

**Transformation rule**

```{math}
f(x) = e^x
```
````
`````

**Example**

```python
from baybe.transformations import ExponentialTransformation

t = ExponentialTransformation()  # nothing to be configured here
```

### LogarithmicTransformation

The {class}`~baybe.transformations.core.LogarithmicTransformation` computes the
natural logarithm of the given input.

`````{grid} 2

````{grid-item}
:columns: auto

![Transforms](../_static/favicon.ico)
````

````{grid-item}
:columns: auto

**Transformation rule**

```{math}
f(x) = \ln(x)
```
````
`````

**Example**

```python
from baybe.transformations import LogarithmicTransformation

t = LogarithmicTransformation()  # nothing to be configured here
```

### PowerTransformation

The {class}`~baybe.transformations.core.PowerTransformation` raises its input to a
specified power.

`````{grid} 2

````{grid-item}
:columns: auto

![Transforms](../_static/favicon.ico)
````

````{grid-item}
:columns: auto

**Transformation rule**

```{math}
f(x) = x^p
```
where $p$ is the exponent specified for the transformation.
````
`````

**Example**

```python
from baybe.transformations import PowerTransformation

t = PowerTransformation(power=2)  # squares the input
```

```{admonition} Integer Exponents
:class: attention
Currently, only integer exponents are supported to avoid obtaining complex numbers when
transforming negative inputs. This may change in the future.
```

### ClampingTransformation

The {class}`~baybe.transformations.core.ClampingTransformation` is used to limit the
range of the input values to a specified interval.

`````{grid} 2

````{grid-item}
:columns: auto

![Transforms](../_static/favicon.ico)
````

````{grid-item}
:columns: auto

**Transformation rule**

```{math}
f(x) =
\begin{cases}
    c_\text{min} & \text{if } x < c_\text{min} \\
    x & \text{if } c_\text{min} \leq x < c_\text{max} \\
    c_\text{max} & \text{if } c_\text{max} \leq x
\end{cases}
```
where $c_\text{min}$ and $c_\text{max}$ are the bounds specified for the transformation.
````
`````

**Example**

```python
from baybe.transformations import ClampingTransformation

t_min = ClampingTransformation(min=10)  # clamps to [10, +inf)
t_max = ClampingTransformation(max=20)  # clamps to (-inf, 20]
t_both = ClampingTransformation(min=10, max=20)  # clamps to [10, 20]
```

### AffineTransformation

The {class}`~baybe.transformations.core.AffineTransformation` applies an affine
transformation to the given input, i.e., it scales and shifts the incoming values.

`````{grid} 2

````{grid-item}
:columns: auto

![Transforms](../_static/favicon.ico)
````

````{grid-item}
:columns: auto

**Transformation rule**

```{math}
f(x) = \begin{cases}
    & = ax + b &\quad \text{by default} \\
    & = a(x + b) &\quad \text{if } \texttt{shift_first=True}
\end{cases}
```
where $a$ is the scaling factor and $b$ is the shift value of the transformation.
````
`````

**Example**

```python
from baybe.transformations import AffineTransformation

t = AffineTransformation(factor=2)  # scales by 2
t = AffineTransformation(shift=3)  # shifts by 3
t = AffineTransformation(factor=2, shift=3)  # scales and *then* shifts
t = AffineTransformation(factor=2, shift=3, shift_first=True)  # shifts and *then* scales
```

### TwoSidedLinearTransformation

The {class}`~baybe.transformations.core.TwoSidedLinearTransformation` is a piecewise
transformation with two linear segments that meet at a midpoint.

`````{grid} 2

````{grid-item}
:columns: auto

![Transforms](../_static/favicon.ico)
````

````{grid-item}
:columns: 6

**Transformation rule**

```{math}
f(x) =
\begin{cases}
    c_\text{left} (x - c_\text{mid}) & \text{if } x < c_\text{mid} \\
    c_\text{right} (x - c_\text{mid}) & \text{if } c_\text{mid} \leq x \\
\end{cases}
```
where $c_\text{left}$ and $c_\text{right}$ are the slopes of the left and right linear
segments, respectively, and $c_\text{mid}$ specifies the midpoint where the two
segments meet.
````
`````

**Example**

```python
from baybe.transformations import TwoSidedLinearTransformation

t = TwoSidedLinearTransformation(left_slope=-1, right_slope=1)  # absolute value
t = TwoSidedLinearTransformation(left_slope=-1, right_slope=0, midpoint=1)  # hinge loss
```

### BellTransformation

The {class}`~baybe.transformations.core.BellTransformation` is pipes the input through a
bell-shaped function (i.e. an **unnormalized** Gaussian), useful for steering the the
input to a specific set point value.

`````{grid} 2

````{grid-item}
:columns: auto

![Transforms](../_static/favicon.ico)
````

````{grid-item}
:columns: 6

**Transformation rule**

```{math}
f(x) = e^{-\frac{(x - c)^2}{2\sigma^2}}
```
where $c$ is the center of the bell and $\sigma$ is a parameter controlling its width.
The latter has the same interpretation as the standard deviation of a Gaussian
distribution except that it does not affect the magnitude of the curve.
````
`````

**Example**

```python
from baybe.transformations import BellTransformation

t = BellTransformation(center=0, sigma=1)  # like an **unnormalized** standard normal distribution
```

### TriangularTransformation

The {class}`~baybe.transformations.core.TriangularTransformation` is a piecewise
linear transformation with the shape of a triangle, useful for steering the the input
to a specific set point value.

`````{grid} 2

````{grid-item}
:columns: auto

![Transforms](../_static/favicon.ico)
````

````{grid-item}
:columns: auto

**Transformation rule**

```{math}
f(x) =
\begin{cases}
    0 & \text{if } x < c_\text{min} \\
    \frac{x - c_\text{min}}{c_\text{peak} - c_\text{min}} & \text{if } c_\text{min} \leq x < c_\text{peak} \\
    \frac{c_\text{max} - x}{c_\text{max} - c_\text{peak}} & \text{if } c_\text{peak} \leq x < c_\text{max} \\
    0 & \text{if } c_\text{max} \leq x
\end{cases}
```
where $c_\text{min}$ and $c_\text{max}$ are the cutoff values of the triangle,
respectively, and $c_\text{peak}$ is its peak location.
````
`````

**Example**

```python
from baybe.transformations import TriangularTransformation  

# Symmetric triangle with peak at 3, reaching zero at 1 and 5
t_sym1 = TriangularTransformation(cutoffs=(1, 5))  
t_sym2 = TriangularTransformation.from_width(peak=3, width=2)
t_sym3 = TriangularTransformation.from_margins(peak=3, margins=(2, 2))
assert t1 == t2 == t3

# Positively skewed triangle with peak at 2 (same cutoffs as above)
t_skew1 = TriangularTransformation(cutoffs=(1, 3), peak=2)
t_skew2 = TriangularTransformation.from_margins(peak=2, margins=(1, 3))
assert t_skew1 == t_skew2
```
