# Search Spaces

The term "search space" refers to the domain of possible values for the parameters that are being optimized during a campaign. A search space represents the space within which BayBE explores and searches for the optimal solution. It is implemented via the [`SearchSpace`](baybe.searchspace.core.SearchSpace) class.

Note that a search space is not necessarily equal to the space of allowed measurements. That is, if configured properly, it is possible to add measurements to a campaign that are not part of the search space. For instance, a numerical parameter with values `1.0`, `2.0`, `5.0` will create a searchspace with these numbers, but you can also add measurements where the parameter has a value of e.g. `2.12`.

In BayBE, a search space is a union of two (potentially empty) subspaces. The [`SubspaceDiscrete`](baybe.searchspace.discrete.SubspaceDiscrete) contains all discrete parameters, while the [`SubspaceContinuous`](baybe.searchspace.continuous.SubspaceContinuous) contains all continuous parameters.

Depending on which of the subspaces are non-empty, a `SearchSpace` has exactly one of the three [`SearchSpaceType`](baybe.searchspace.core.SearchSpaceType)'s:

|`SubspaceDiscrete`|`SubspaceContinuous`|[`SearchSpaceType`](baybe.searchspace.core.SearchSpaceType)|
|------------------|--------------------|-----------------|
|Non-empty|Empty|[`SearchSpaceType.DISCRETE`](baybe.searchspace.core.SearchSpaceType.DISCRETE)|
|Empty|Non-Empty|[`SearchSpaceType.CONTINUOUS`](baybe.searchspace.core.SearchSpaceType.CONTINUOUS)|
|Non-Empty|Non-empty|[`SearchSpaceType.HYBRID`](baybe.searchspace.core.SearchSpaceType.HYBRID)|


## Discrete Subspaces

The `SubspaceDiscrete` contains all the discrete parameters of a `SearchSpace`. There are different ways of constructing this subspace.

### Building from the Product of Parameter Values

The method [`SearchSpace.from_product`](baybe.searchspace.discrete.SubspaceDiscrete.from_product) constructs the full cartesian product of the provided parameters:

```python
from baybe.parameters import NumericalDiscreteParameter, CategoricalParameter
from baybe.searchspace import SubspaceDiscrete

parameters = [
    NumericalDiscreteParameter(name="x0", values=[1, 2, 3]),
    NumericalDiscreteParameter(name="x1", values=[4, 5, 6]),
    CategoricalParameter(name="Speed", values=["slow", "normal", "fast"]),
]
subspace = SubspaceDiscrete.from_product(parameters=parameters)
```

In this example, `subspace` has a total of 27 different parameter configuration.

~~~
      x0   x1   Speed
 0   1.0  4.0    slow
 1   1.0  4.0  normal
 2   1.0  4.0    fast
 ..  ...  ...     ...
 24  3.0  6.0    slow
 25  3.0  6.0  normal
 26  3.0  6.0    fast
  
  [27 rows x 3 columns]
~~~

### Constructing from a Dataframe

[`SubspaceDiscrete.from_dataframe`](baybe.searchspace.discrete.SubspaceDiscrete.from_dataframe) constructs a discrete subspace from a given dataframe.
By default, this method tries to infer the data column as as a [`NumericalDiscreteParameter`](baybe.parameters.numerical.NumericalDiscreteParameter) and uses [`CategoricalParameter`](baybe.parameters.categorical.CategoricalParameter) as fallback.
However, it is possible to change this behavior by using the optional `parameters` keyword.
This list informs `from_dataframe` about the parameters and the types of parameters that should be used.
In particular, it is necessary to provide such a list if there are non-numerical parameters that should not be interpreted as categorical parameters.

```python
import pandas as pd

df = pd.DataFrame(
    {
        "x0": [2, 3, 3],
        "x1": [5, 4, 6],
        "x2": [9, 7, 9],
    }
)
subspace = SubspaceDiscrete.from_dataframe(df)
```

~~~
 Discrete Parameters
   Name                        Type  Num_Values Encoding
 0   x0  NumericalDiscreteParameter           2     None
 1   x1  NumericalDiscreteParameter           3     None
 2   x2  NumericalDiscreteParameter           2     None
~~~

### Creating a Simplex-Bound Discrete Subspace

[`SubspaceDiscrete.from_simplex`](baybe.searchspace.discrete.SubspaceDiscrete.from_simplex) can be used to efficiently create a discrete search space (or discrete subspace) that is restricted by a simplex constraint, limiting the maximum sum of the parameters per dimension.
This method uses a shortcut that removes invalid candidates already during the creation of parameter combinations and avoids to first create the full product space before filtering it.

In the following example, a naive construction of the subspace would first construct the full product space, containing 25 points, although only 15 points are actually part of the simplex.

```python
parameters = [
    NumericalDiscreteParameter(name="p1", values=[0, 0.25, 0.5, 0.75, 1]),
    NumericalDiscreteParameter(name="p2", values=[0, 0.25, 0.5, 0.75, 1]),
]
subspace = SubspaceDiscrete.from_simplex(max_sum=1.0, simplex_parameters=parameters)
```

~~~
       p1    p2
 0   0.00  0.00
 1   0.00  0.25
 2   0.00  0.50
 ..   ...   ...
 12  0.75  0.00
 13  0.75  0.25
 14  1.00  0.00
 
 [15 rows x 2 columns]
~~~

Note that it is also possible to provide additional parameters that then enter in the form of a Cartesian product.
These can be provided via the keyword `product_parameters`.

(DATA_REPRESENTATION)=
### Representation of Data within Discrete Subspaces

Internally, discrete subspaces are represented by two dataframes, the *experimental* and the *computational* representation.

The experimental representation (`exp_rep`) contains all parameters as they were provided upon the construction of the search space and viewed by the experimenter. The computational representation (`comp_rep`) contains a representation of parameters that is actually used for the internal calculation.

In particular, the computational representation contains no more labels or constant columns. This happens e.g. for [`SubstanceParameter`](baybe.parameters.substance.SubstanceParameter) or [`CategoricalParameter`](baybe.parameters.categorical.CategoricalParameter). Further, note that the shape of the computational representation can also change depending on the chosen encoding.

The following example demonstrates the difference:
```python
from baybe.parameters import NumericalDiscreteParameter, CategoricalParameter

speed = CategoricalParameter("Speed", values=["slow", "normal", "fast"], encoding="OHE")
temperature = NumericalDiscreteParameter(name="Temperature", values=[90, 105])

subspace = SubspaceDiscrete.from_product(parameters=[speed, temperature])
```

~~~
  Experimental Representation
      Speed  Temperature
  0    slow         90.0
  1    slow        105.0
  2  normal         90.0
  3  normal        105.0
  4    fast         90.0
  5    fast        105.0

  Computational Representation
     Speed_slow  Speed_normal  Speed_fast  Temperature
  0           1             0           0         90.0
  1           1             0           0        105.0
  2           0             1           0         90.0
  3           0             1           0        105.0
  4           0             0           1         90.0
  5           0             0           1        105.0
~~~

### Metadata

```{warning}
Although possible, it is not intended to manually change the metadata. The metadata is maintained internally, and there is a risk involved with manipulating it manually.
Consequently, we advise to only change the metadata manually if you are certain about it.
```

Discrete subspaces keep track of the recommendations that were made during a campaign.
The information is stored in a [`metadata`](baybe.searchspace.discrete.SubspaceDiscrete.metadata) field, and it is possible to manually modify `metadata` to influence the behavior of the corresponding `campaign` exploring the space. In particular, by manually changing the values of `metadata["dont_recommend"]`, it is possible to prevent certain points of the subspace from being recommended.

## Continuous Subspaces

The `SubspaceContinuous` contains all the continuous parameters of a `SearchSpace`. There are different ways of constructing this subspace.

### Using Explicit Bounds

The [`SubspaceContinuous.from_bounds`](baybe.searchspace.continuous.SubspaceContinuous.from_bounds) method can be used to easily create a subspace representing a hyperrectangle.

```python
from baybe.searchspace import SubspaceContinuous

bounds = pd.DataFrame({"param1": [0, 1], "param2": [-1, 1]})
subspace = continuous = SubspaceContinuous.from_bounds(bounds)
```

~~~
 Continuous Parameters
      Name                          Type  Lower_Bound  Upper_Bound
 0  param1  NumericalContinuousParameter          0.0          1.0
 1  param2  NumericalContinuousParameter         -1.0          1.0
~~~

### Constructing from a Dataframe

Similar to discrete subspaces, continuous spaces can also be constructed using [`SubspaceContinuous.from_dataframe`](baybe.searchspace.continuous.SubspaceContinuous.from_dataframe).
However, when using this method to create a continuous space, it will create the smallest axis-aligned hyperrectangle-shaped continuous subspace that contains the points specified in the given dataframe.

```python
from baybe.parameters import NumericalContinuousParameter
from baybe.searchspace.continuous import SubspaceContinuous

points = pd.DataFrame(
    {
        "param1": [0, 1, 2],
        "param2": [-1, 0, 1],
    }
)
subspace = SubspaceContinuous.from_dataframe(df=points)
```

As for discrete subspaces, this method automatically infers the parameter types but can be provided with an optional list `parameters`.

~~~
 Continuous Parameters
      Name                          Type  Lower_Bound  Upper_Bound
 0  param1  NumericalContinuousParameter          0.0          2.0
 1  param2  NumericalContinuousParameter         -1.0          1.0
~~~

## Constructing Full Search Spaces

There are several methods available for creating full search spaces.

### From the Default Constructor 

It is possible to construct a search space by simply using the default constructor of the `SearchSpace` class.
The required parameters are derived from the `__init__` function of that class.
In the simplest setting, it is sufficient to provide a single subspace for creating either a discrete or continuous search, or provide two subspaces for creating a hybrid search space.

~~~python
searchspace = SearchSpace(discrete=discrete_subspace, continuous=continuous_subspace)
~~~

While this constructor is the default choice, it might not be the most convenient.
Consequently, other constructors are available.

### Building from the Product of Parameter Values

The function [`SearchSpace.from_product`](baybe.searchspace.core.SearchSpace.from_product) is analog to the corresponding function available for `SubspaceDiscrete`, but allows the parameter list to contain both discrete and continuous parameters.

### Constructing from a Dataframe

[`SearchSpace.from_dataframe`](baybe.searchspace.core.SearchSpace.from_dataframe) constructs a search space from a given dataframe.
Due to the ambiguity between discrete and numerical parameter choices, this function requires a parameters list with pre-defined parameter objects, unlike its subspace counterparts, where this list is optional.

```python
from baybe.searchspace import SearchSpace

params = [
    NumericalDiscreteParameter(name="x0", values=[1, 2, 3]),
    NumericalDiscreteParameter(name="x1", values=[4, 5, 6]),
    NumericalContinuousParameter(name="x2", bounds=(6, 9)),
]

df = pd.DataFrame(
    {
        "x0": [2, 3],
        "x1": [5, 4],
        "x2": [9, 7],
    }
)
searchspace = SearchSpace.from_dataframe(df=df, parameters=params)
```

Since one of the provided parameters is continuous, this creates a hybrid space.
The following shows *all* information that are available to the user for this space:

~~~
Search Space
         
 Search Space Type: HYBRID
         
 Discrete Search Space
              
  Discrete Parameters
    Name                        Type  Num_Values Encoding
  0   x0  NumericalDiscreteParameter           3     None
  1   x1  NumericalDiscreteParameter           3     None
              
  Experimental Representation
     x0  x1   
  0   2   5
  1   3   4
  
  Metadata:
  was_recommended: 0/2
  was_measured: 0/2
  dont_recommend: 0/2
              
  Constraints
  Empty DataFrame
  Columns: []
  Index: []
              
  Computational Representation
     x0  x1   
  0   2   5
  1   3   4
         
 Continuous Search Space
              
  Continuous Parameters
    Name                          Type  Lower_Bound  Upper_Bound
  0   x2  NumericalContinuousParameter          6.0          9.0
              
  List of Linear Equality Constraints
  Empty DataFrame
  Columns: []
  Index: []
              
  List of Linear Inequality Constraints
  Empty DataFrame
  Columns: []
  Index: []
~~~

## Restricting Search Spaces Using Constraints

Most constructors for both subspaces and search spaces support the optional keyword argument `constraints` to provide a list of [`Constraint`](baybe.constraints.base.Constraint) objects. 
When constructing full search spaces, the type of each constraint is checked, and the consequently applied to the corresponding subspace.

~~~python
constraints = [...]
# Using one example constructor here
searchspace = SearchSpace.from_product(parameters=parameters, constraints=constraints)
~~~

For a more in-depth discussion of constraints, we refer to the corresponding  [user guide](../../userguide/constraints) for more details.