# Constraints
Experimental campaigns often have naturally arising constraints on the parameters and
their combinations. Such constraints could for example be:
* When optimizing a mixture, the relative concentrations of the used ingredients must
  add up to 1.0.
* For chemical reactions, a reagent might be incompatible with high temperatures, hence
  these combinations must be excluded.
* Certain settings are dependent on other parameters, e.g. a set of parameters only
  becomes relevant if another parameter called `"Switch"` has the value `"on"`.

Similar to parameters, BayBE distinguishes two families of constraints, derived from the 
abstract [`Constraint`](baybe.constraints.base.Constraint) class: discrete and
continuous constraints ([`DiscreteConstraint`](baybe.constraints.base.DiscreteConstraint), 
[`ContinuousConstraint`](baybe.constraints.base.ContinuousConstraint)).
A constraint is called discrete/continuous if it operates on a set of exclusively
discrete/continuous parameters.

```{admonition} Hybrid constraints
:class: note
Currently, BayBE does not support hybrid constraints, that is, constraints which
operate on a mixed set of discrete and continuous parameters. If such a constraint is
necessary, it is possible to rephrase the parametrization so that the parameter set
is exclusively discrete or continuous in most cases.
```

## Continuous Constraints

```{warning}
Not all surrogate models are able to treat continuous constraints. In such situations
the constraints are currently silently ignored.
```   

(CLC)=
### ContinuousLinearConstraint
The [`ContinuousLinearConstraint`](baybe.constraints.continuous.ContinuousLinearConstraint)
asserts that the following kind of equations are true (up to numerical rounding errors):

$$
\sum_{i} x_i \cdot c_i = \text{rhs} \\
\sum_{i} x_i \cdot c_i >= \text{rhs} \\
\sum_{i} x_i \cdot c_i <= \text{rhs}
$$

where $x_i$ is the value of the $i$'th parameter affected by the constraint,
$c_i$ is the coefficient for that parameter, and $\text{rhs}$ is a user-chosen number.
The (in)equality type is defined by the `operator` keyword.

As an example, let's assume we have three parameters named `x_1`, `x_2` and
`x_3`, which describe the relative concentrations in a mixture campaign.
The constraint assuring that they always sum up to 1.0 would look like this:

```python
from baybe.constraints import ContinuousLinearConstraint

ContinuousLinearConstraint(
    parameters=["x_1", "x_2", "x_3"],  # these parameters must exist in the search space
    operator="=",
    coefficients=[1.0, 1.0, 1.0],
    rhs=1.0,
)
```

Let us amend the example from above and assume that there is always a fourth component
to the mixture that serves as a "filler". In such a case, we might want to ensure that
the first three components only make up to 80% of the mixture.
The following constraint would achieve this:

```python
from baybe.constraints import ContinuousLinearConstraint

ContinuousLinearConstraint(
    parameters=["x_1", "x_2", "x_3"],
    operator="<=",
    coefficients=[1.0, 1.0, 1.0],
    rhs=0.8,
)
```

A more detailed example can be found
[here](../../examples/Constraints_Continuous/linear_constraints).

## Conditions
Conditions are elements used within discrete constraints.
While discrete constraints can operate on one or multiple parameters, a condition
always describes the relation of a single parameter to its possible values.
It is through chaining several conditions in constraints that we can build complex
logical expressions for them.

### ThresholdCondition
For numerical parameters, we might want to select a certain range, which can be
achieved with a [`ThresholdCondition`](baybe.constraints.conditions.ThresholdCondition):
```python
from baybe.constraints import ThresholdCondition

ThresholdCondition(  # will select all values above 150
    threshold=150,
    operator=">",
)
```

### SubSelectionCondition
In case a specific subset of values needs to be selected, it can be done with the
[`SubSelectionCondition`](baybe.constraints.conditions.SubSelectionCondition):
```python
from baybe.constraints import SubSelectionCondition

SubSelectionCondition(  # will select two solvents identified by their labels
    selection=["Ethanol", "DMF"]
)
```

## Discrete Constraints
Discrete constraints currently do not affect the optimization process directly.
Instead, they act as a filter on the search space.
For instance, a search space created via [`from_product`](baybe.searchspace.core.SearchSpace.from_product) 
might include invalid combinations, which can be removed again by applying constraints.

Discrete constraints have in common that they operate on one or more parameters,
identified by the `parameters` member, which expects a list of parameter names as
strings.
All of these parameters must be present in the search space specification.

### DiscreteExcludeConstraint
The [`DiscreteExcludeConstraint`](baybe.constraints.discrete.DiscreteExcludeConstraint)
constraint simply removes a set of search space elements, according to its
specifications.

The following example would exclude entries where "Ethanol" and "DMF" are combined with
temperatures above 150, which might be due to their chemical instability at those
temperatures:
```python
from baybe.constraints import (
    DiscreteExcludeConstraint,
    ThresholdCondition,
    SubSelectionCondition,
)

DiscreteExcludeConstraint(
    parameters=["Temperature", "Solvent"],  # names of the affected parameters
    combiner="AND",  # specifies how the conditions are logically combined
    conditions=[  # requires one condition for each entry in parameters
        ThresholdCondition(threshold=150, operator=">"),
        SubSelectionCondition(selection=["Ethanol", "DMF"]),
    ],
)
```

A more detailed example can be found
[here](../../examples/Constraints_Discrete/exclusion_constraints).

### DiscreteSumConstraint and DiscreteProductConstraint
[`DiscreteSumConstraint`](baybe.constraints.discrete.DiscreteSumConstraint)
and [`DiscreteProductConstraint`](baybe.constraints.discrete.DiscreteProductConstraint)
impose conditions on sums or products of numerical parameters.
[In the first example from `ContinuousLinearConstraint`](#CLC), we
had three continuous parameters `x_1`, `x_2` and `x_3`, which needed to sum
up to 1.0.
If these parameters were instead discrete, the corresponding constraint would look like:
```python
from baybe.constraints import DiscreteSumConstraint, ThresholdCondition

DiscreteSumConstraint(
    parameters=["x_1", "x_2", "x_3"],
    condition=ThresholdCondition(  # set condition that should apply to the sum
        threshold=1.0,
        operator="=",
        tolerance=0.001,  # optional; here, everything between 0.999 and 1.001 would also be considered valid
    ),
)
```

An end to end example can be found [here](../../examples/Constraints_Discrete/prodsum_constraints).

### DiscreteNoLabelDuplicatesConstraint
Sometimes, duplicated labels in several parameters are undesirable.
Consider an example with two solvents that describe different mixture
components.
These might have the exact same or overlapping sets of possible values, e.g.
`["Water", "THF", "Octanol"]`.
It would not necessarily be reasonable to allow values in which both solvents show the
same label/component.
We can exclude such occurrences with the
[`DiscreteNoLabelDuplicatesConstraint`](baybe.constraints.discrete.DiscreteNoLabelDuplicatesConstraint):

```python
from baybe.constraints import DiscreteNoLabelDuplicatesConstraint

DiscreteNoLabelDuplicatesConstraint(parameters=["Solvent_1", "Solvent_2"])
```

Without this constraint, combinations like below would be possible:

|   | Solvent_1 | Solvent_2 | With DiscreteNoLabelDuplicatesConstraint |
|---|-----------|-----------|------------------------------------------|
| 1 | Water     | Water     | would be excluded                        |
| 2 | THF       | Water     |                                          |
| 3 | Octanol   | Octanol   | would be excluded                        |

The usage of `DiscreteNoLabelDuplicatesConstraint` is part of the
[example on mixtures](../../examples/Constraints_Discrete/mixture_constraints).

### DiscreteLinkedParametersConstraint
The [`DiscreteLinkedParametersConstraint`](baybe.constraints.discrete.DiscreteLinkedParametersConstraint)
is, in a sense, the opposite of the
[`DiscreteNoLabelDuplicatesConstraint`](baybe.constraints.discrete.DiscreteNoLabelDuplicatesConstraint).
It will ensure that **only** entries with duplicated labels are present.
This can be useful, for instance, in situations where we have one parameter but would
like to include it with several encodings:
```python
from baybe.parameters import SubstanceParameter
from baybe.constraints import DiscreteLinkedParametersConstraint

dict_solvents = {"Water": "O", "THF": "C1CCOC1", "Octanol": "CCCCCCCCO"}
solvent_encoding1 = SubstanceParameter(
    name="Solvent_RDKIT_enc",
    data=dict_solvents,
    encoding="RDKIT",
)
solvent_encoding2 = SubstanceParameter(
    name="Solvent_MORDRED_enc",
    data=dict_solvents,
    encoding="MORDRED",
)
DiscreteLinkedParametersConstraint(
    parameters=["Solvent_RDKIT_enc", "Solvent_MORDRED_enc"]
)
```

|   | Solvent_RDKIT_enc | Solvent_MORDRED_enc | With DiscreteLinkedParametersConstraint |
|---|-------------------|---------------------|-----------------------------------------|
| 1 | Water             | Water               |                                         |
| 2 | THF               | Water               | would be excluded                       |
| 3 | Octanol           | Octanol             |                                         |

### DiscreteDependenciesConstraint
A dependency is a situation where parameters depend on other parameters.
Let's say an experimental setup has a parameter called `"Switch"`, which turns on
pieces of equipment that are optional.
This means the other parameters (called `affected_parameters`) are only relevant if
the switch parameter has the value `"on"`. If the switch is `"off"`, the affected 
parameters are irrelevant.

You can specify such a dependency with the
[`DiscreteDependenciesConstraint`](baybe.constraints.discrete.DiscreteDependenciesConstraint)
, which requires:
1) A list `parameters` with the names of the parameters upon which others depend.
2) A list `conditions`, specifying the values of the corresponding entries in
   `parameters` that "activate" the dependent parameters.
3) A list of lists, each containing the `affected_parameters`, which become relevant
   only if the corresponding entry in `parameters` is active as specified by the 
   entry in `conditions`.

Internally, BayBE drops elements from the `SearchSpace` where affected parameters are
irrelevant. Since in our example `"off"` is still a valid value for the switch, the
`SearchSpace` will still retain **one** configuration for that setting, showing arbitrary
values for the `affected_parameters` (which can be ignored).

(DDC)=
```{important}
BayBE requires that all dependencies are declared in a single
`DiscreteDependenciesConstraint`. Creating a `SearchSpace` from multiple
`DiscreteDependenciesConstraint`'s will throw a validation error.
```

In the example below, we mimic a situation where there are two switches and each switch
activates two other parameters that are only relevant if the first switch is `"on"` / the
second switch is set to `"right"`, respectively.
```python
from baybe.constraints import DiscreteDependenciesConstraint, SubSelectionCondition

DiscreteDependenciesConstraint(
    parameters=["Switch_1", "Switch_2"],  # the two parameters upon which others depend
    conditions=[
        SubSelectionCondition(
            # values of Switch_1 that activate the affected parameters
            selection=["on"]
        ),
        SubSelectionCondition(
            # values of Switch_2 that activate the affected parameters
            selection=["right"]
        ),
    ],
    affected_parameters=[
        ["Solvent", "Fraction"],  # parameters affected by Switch_1
        ["Frame_1", "Frame_2"],  # parameters affected by Switch_2
    ],
)
```

An end to end example can be found [here](../../examples/Constraints_Discrete/dependency_constraints).

### DiscretePermutationInvarianceConstraint
Permutation invariance, enabled by the 
[`DiscretePermutationInvarianceConstraint`](baybe.constraints.discrete.DiscretePermutationInvarianceConstraint)
, is a property where combinations of values of multiple
parameters do not depend on their order due to some symmetry in the experiment.
Suppose we create a mixture containing up to three solvents, i.e. parameters
"Solvent_1", "Solvent_2", "Solvent_3".
In this situation, all combinations from the following table would be equivalent,
hence the `SearchSpace` should effectively only contain one of them.

|   | Solvent_1    | Solvent_2    | Solvent_3    |
|---|--------------|--------------|--------------|
| 1 | Substance_43 | Substance_3  | Substance_12 |
| 2 | Substance_43 | Substance_12 | Substance_3  |
| 3 | Substance_3  | Substance_12 | Substance_43 |
| 4 | Substance_3  | Substance_43 | Substance_12 |
| 5 | Substance_12 | Substance_43 | Substance_3  |
| 6 | Substance_12 | Substance_3  | Substance_43 |

```{note}
Complex properties such as permutation invariance not only affect the search space but
should ideally also constrain the surrogate model. For instance, the kernels in a
Gaussian process can be made permutation-invariant to reflect this constraint, which
generally results in a better learning curve. Note that at this stage no
surrogate model provided by BayBE takes care of these invariances. This means the
invariance is ignored during model fitting and these models do not benefit
from a priori known constraints and invariances between parameters. However, generally,
the optimization will still work. We are in the process of enabling this as new feature,
but in the meantime the user can introduce their own
[custom surrogate model](../../examples/Custom_Surrogates/Custom_Surrogates)
to include these.
```

Let's add to the mixture example the fact that not only the choice of substance but also
their relative mixture fractions are parameters, i.e. "Fraction_1", "Fraction_2" and
"Fraction_3".
This also implies that the solvent parameters depend on their corresponding
fraction being `> 0.0`, because in the case `== 0.0` the choice of solvent is
irrelevant. This models a scenario that allows "up to, but not necessarily,
three solvents".

```{important}
If some of the `parameters` of the `DiscretePermutationInvarianceConstraint` are
dependent on other parameters, we require that the dependencies are provided as a
`DiscreteDependenciesConstraint` to the `dependencies` argument of the
`DiscretePermutationInvarianceConstraint`. This
`DiscreteDependenciesConstraint` will not count towards the maximum limit of one
`DiscreteDependenciesConstraint` discussed [here](#DDC).
```

The `DiscretePermutationInvarianceConstraint` below applies to our example and
removes permutation-invariant combinations of solvents that have additional
dependencies as well:

```python
from baybe.constraints import (
    DiscretePermutationInvarianceConstraint,
    DiscreteDependenciesConstraint,
    ThresholdCondition,
)

DiscretePermutationInvarianceConstraint(
    parameters=["Solvent_1", "Solvent_2", "Solvent_3"],
    # `dependencies` is optional; it is only required if some of the permutation
    # invariant entries in `parameters` have dependencies on other parameters
    dependencies=DiscreteDependenciesConstraint(
        parameters=["Fraction_1", "Fraction_2", "Fraction_3"],
        conditions=[
            ThresholdCondition(threshold=0.0, operator=">"),
            ThresholdCondition(threshold=0.0, operator=">"),
            ThresholdCondition(threshold=0.0, operator=">"),
        ],
        affected_parameters=[["Solvent_1"], ["Solvent_2"], ["Solvent_3"]],
    ),
)
```

The usage of `DiscretePermutationInvarianceConstraint` is also part of the
[example on mixtures](../../examples/Constraints_Discrete/mixture_constraints).

### DiscreteCustomConstraint
With a [`DiscreteCustomConstraint`](baybe.constraints.discrete.DiscreteCustomConstraint) 
constraint, you can specify a completely custom filter:

```python
import pandas as pd
import numpy as np
from baybe.constraints import DiscreteCustomConstraint


def custom_filter(df: pd.DataFrame) -> pd.Series:  # this signature is required
    """
    In this example, we exclude entries where the square root of the
    temperature times the cubed pressure are larger than 5.6.
    """
    mask_good = np.sqrt(df["Temperature"]) * np.power(df["Pressure"], 3) <= 5.6

    return mask_good


DiscreteCustomConstraint(
    parameters=[  # the custom function will have access to these variables
        "Pressure",
        "Temperature",
    ],
    validator=custom_filter,
)
```

Find a detailed example [here](../../examples/Constraints_Discrete/custom_constraints).

```{warning}
Due to the arbitrary nature of code and dependencies that can be used in the
`DiscreteCustomConstraint`, de-/serializability cannot be guaranteed. As a consequence,
using a `DiscreteCustomConstraint` results in an error if you attempt to serialize
the corresponding object or higher-level objects containing it.
```