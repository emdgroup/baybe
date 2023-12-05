# Constraints
Experimental campaigns often have naturally arising constraints on the parameters and 
their combinations. Such constraints could for example be:
* When optimizing a mixture, the relative concentrations of the used ingredients must 
  add up to 1.0.
* For chemical reactions, a reagent might be incompatible with high temperatures, hence 
  these combinations must be excluded.
* Certain settings are dependent on other parameters, e.g. a set of parameters only 
  becomes relevant if another parameter called "Switch" has the value "on".

Similar to parameters, BayBE distinguishes two families of constraints: discrete and 
continuous constraints. 
A constraint is called discrete / continuous if it operates on a set of exclusively 
discrete / continuous parameters.

```{note}
Currently, BayBE does not support hybrid constraints, that is, constraints which 
operate on a mixed set of discrete and continuous parameters. If such a constraint is 
necessary in almost all cases it is possible to rephrase the parametrization so that 
the parameter set is exclusively discrete or continuous.
```

## Continuous Constraints

```{warning}
Not all surrogate models are able to treat continuous constraints. In such situations 
the constraints are currently silently ignored.
```   

(CLEQ)=
### ``ContinuousLinearEqualityConstraint``
This linear constraint asserts that the following equation is true (up to numerical 
rounding errors): 

$$
\sum_{i} x_i \cdot c_i = \text{rhs}
$$

where $x_i$ is the value of the $i$'th parameter affected by the constraint 
and $c_i$ is the coefficient for that parameter. $\text{rhs}$ is a user-chosen number.

As an example we assume we have three parameters named ``x_1``, ``x_2`` and 
``x_3``, which describe the relative concentrations in a mixture campaign.
The constraint assuring that they always sum up to 1.0 would look like this:
```python
from baybe.constraints import ContinuousLinearEqualityConstraint

ContinuousLinearEqualityConstraint(
    parameters = ["x_1", "x_2", "x_3"], # these parameters must exist in the search space
    coefficients = [1.0, 1.0, 1.0], 
    rhs = 1.0
)
```

### ``ContinuousLinearInequalityConstraint``
This linear constraint asserts that the following equation is true (up to numerical 
rounding errors):

$$
\sum_{i} x_i \cdot c_i >= \text{rhs}
$$

where $x_i$ is the value of the $i$'th parameter affected by the constraint, 
$c_i$ is the coefficient for that parameter. $\text{rhs}$ is a user-chosen number.

```{info}
You can specify a constraint involving ``<=`` instead of ``>=`` by multiplying 
both sides, i.e. the coefficients and rhs, by -1.
```

Let us amend the example from 
[`ContinuousLinearEqualityConstraint`](#CLEQ) and assume 
that there is always a fourth component to the mixture which serves as a "filler".
In such a case we might want to ensure that the first three components only make up to 
80% of the mixture. 
The following constraint would achieve this:
```python
from baybe.constraints import ContinuousLinearInequalityConstraint

ContinuousLinearInequalityConstraint(
    parameters = ["x_1", "x_2", "x_3"], # these parameters must exist in the search space
    coefficients = [-1.0, -1.0, -1.0], 
    rhs = -0.8 # coefficients and rhs are negated because we model a `<=` constraint
),
```

## Conditions
Conditions are elements used within discrete constraints. 
While discrete constraints can operate on one or multiple parameters, a condition 
always describes the relation of a single parameter to its possible values.
It is through chaining several conditions in constraints that we can build complex 
logical expressions for them.

### ``ThresholdCondition``
For numerical parameters, we might want to select a certain range, which can be 
achieved with ``ThresholdCondition``:
```python
from baybe.constraints import ThresholdCondition

ThresholdCondition( # will select all values above 150
    threshold = 150, 
    operator = ">",
    tolerance = 0.2 # optional, with this 149.82 would still be valid
)
```

### ``SubSelectionCondition``
In case a specific subset of values needs to be selected, it can be done with the 
``SubSelectionCondition``:
```python
from baybe.constraints import SubSelectionCondition

SubSelectionCondition( # will select two solvents identified by their labels
    selection = ["Solvent A", "Solvent C"]
)
```

## Discrete Constraints
Discrete constraints currently do not affect the optimization process directly. 
Instead, they act as a filter on the search space.
For instance, a search space created via ``from_product`` might include invalid 
combinations, which can be removed again by constraints.

Discrete constraints have in common that they operate on one or more parameters, 
identified by the ``parameters`` member, which expects a list of parameter names as 
strings.
All of these parameters must be present in the campaign specification.

### ``DiscreteExcludeConstraint``
This constraint simply removes a set of search space entries, according to its 
specifications.

This example would exclude entries where "Solvent A" and "Solvent C" are combined with 
temperatures above 150, which might be due to their chemical instability at those 
temperatures:
```python
from baybe.constraints import DiscreteExcludeConstraint, ThresholdCondition, SubSelectionCondition

DiscreteExcludeConstraint(
    parameters = ["Temperature", "Solvent"], # names of the affected parameters
    combiner = "AND", # specifies how the conditions are logically combined
    conditions = [ # requires one condition for each entry in parameters
        ThresholdCondition(threshold = 150, operator = ">"),
        SubSelectionCondition(selection = ["Solvent A", "Solvent B"]),
    ]
)
```

### ``DiscreteSumConstraint`` and ``DiscreteProductConstraint``
These constraints constrain sums or products of numerical parameters. In the example 
from [``ContinuousLinearEqualityConstraint``](#CLEQ) we 
had three continuous parameters ``x_1``, ``x_2`` and ``x_3`` which needed to sum 
up to 1.0.
If these parameters were instead discrete, the corresponding constraint would look like:
```python
from baybe.constraints import DiscreteSumConstraint, ThresholdCondition

DiscreteSumConstraint(
    parameters = ["x_1", "x_2", "x_3"],
    condition = ThresholdCondition( # set condition that should apply to the sum
        threshold = 1.0, 
        operator = "=", 
        tolerance = 0.001) # optional, with this 0.999 or 1.001 would also be valid
)
```

### ``DiscreteNoLabelDuplicatesConstraint``
Sometimes duplicated labels in several parameters are undesirable.
Consider an example where we have two solvents which describe different mixture 
components.
These might have the exact same or overlapping sets of possible values, e.g. 
``["Water", "THF", "Octanol"]``.
It would not necessarily be reasonable to allow values in which both solvents show the 
same label/component.
We can exclude such occurrences with the ``DiscreteNoLabelDuplicatesConstraint``:

```python
from baybe.constraints import  DiscreteNoLabelDuplicatesConstraint

DiscreteNoLabelDuplicatesConstraint(
    parameters=["Solvent1", "Solvent2"]
)
```

Without this constraint, combinations like below would be possible:

|   | Solvent1 | Solvent2 | With DiscreteNoLabelDuplicatesConstraint |
|---|----------|----------|------------------------------------------|
| 1 | Water    | Water    | would be excluded                        |
| 2 | THF      | Water    |                                          |
| 3 | Octanol  | Octanol  | would be excluded                        |

### ``DiscreteLinkedParametersConstraint``
The ``DiscreteLinkedParametersConstraint`` in a sense is the opposite of the 
``DiscreteNoLabelDuplicatesConstraint``. 
It will ensure that **only** entries with duplicated labels are present.
This can be useful for instance in a situation where we have one parameter, but would
like to include it with several encodings:
```python
from baybe.parameters import SubstanceParameter
from baybe.constraints import DiscreteLinkedParametersConstraint

dict_solvents = {"Water": "O", "THF": "C1CCOC1", "Octanol": "CCCCCCCCO"}
solvent_encoding1 = SubstanceParameter(
    name = 'Solvent_RDKIT_enc',
    data = dict_solvents,
    encoding = "RDKIT",
)
solvent_encoding2 = SubstanceParameter(
    name = 'Solvent_MORDRED_enc',
    data = dict_solvents,
    encoding = "MORDRED",
)
DiscreteLinkedParametersConstraint(
    parameters = ["Solvent_RDKIT_enc", "Solvent_MORDRED_enc"]
)
```

|   | Solvent_RDKIT_enc | Solvent_MORDRED_enc | With DiscreteLinkedParametersConstraint |
|---|-------------------|---------------------|-----------------------------------------|
| 1 | Water             | Water               |                                         |
| 2 | THF               | Water               | would be excluded                       |
| 3 | Octanol           | Octanol             |                                         |

### ``DiscreteDependenciesConstraint``
Content coming soon...

### ``DiscretePermutationInvarianceConstraint``
Content coming soon...

### ``DiscreteCustomConstraint``
With this constraint you can specify a completely custom filter:

```python
import pandas as pd
import numpy as np
from baybe.constraints import DiscreteCustomConstraint

def custom_filter(series: pd.Series) -> bool: # this signature is required
    """
    In this example we exclude entries where the square root of the 
    temperature times the cubed pressure are larger than 5.6 by returning False in 
    those cases, and True otherwise.
    """
    if np.sqrt(series.Temperature) * np.power(series.Pressure,3) > 5.6:
        return False
    return True

DiscreteCustomConstraint(
    parameters = ["Pressure", "Temperature"], # the custom function will have access to these variables 
    validator = custom_filter
)
```

```{warning}
Due to the arbitrary nature of code and dependencies that can be used in the 
``DiscreteCustomConstraint``, de-/serializability cannot be guaranteed. As a result, 
using a ``DiscreteCustomConstraint`` results in an error if you attempt to serialize 
the corresponding ``Campaign``.
```
