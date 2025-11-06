# Checklist for Designing BayBE Optimization Campaigns

This checklist collects **common questions that you need to ask yourself** when designing a BayBE optimization campaign. It also provides documentation references that will help you with the corresponding setup. 

Note that this is not a comprehensive guide of all BayBE's functionalities, but rather a "quick start" meant to help you with the most basic design principles.

## Defining Targets

```{dropdown} Should the target value be maximized, minimized, or be matched to a specific value?

Specify this when
[defining the target](userguide/targets.html#numericaltarget).
```

```{dropdown} Should multiple target be optimized simultaneously?

See how to use 
[multi-target objectives](userguide/objectives.html).
```

## Defining Parameter Search Space

```{dropdown} Are only some parameter values of interest/possible?

See how to exclude some 
[parameter values](userguide/getting_recommendations.html#excluding-configurations) 
from being recommended, such as by defining
[bounds for continuous parameters](userguide/parameters.html#numericalcontinuousparameter)
or [active values for discrete parameters](userguide/parameters.html#discrete-parameters).
```

```{dropdown} Are only some parameter combinations of interest/possible?

See how to exclude some 
parameter combinations from being considered by using 
[constraints](userguide/constraints.html) or
[constrained searchspaces](userguide/searchspace.html#creating-a-simplex-bound-discrete-subspace).
Alternatively, if the aim is to use only a few specific parameter configurations the search space can be created from a 
[dataframe](userguide/searchspace.html#id3) 
rather than from the product of all possible parameter combinations.
```

```{dropdown} Are some parameters non-numeric or allow only discrete numbers?

Use [discrete](userguide/parameters.html#discrete-parameters)
rather than [continuous](userguide/parameters.html#continuous-parameters) parameters.
```

```{dropdown} Is it possible to encode discrete parameters based on domain knowledge to capture relationships between categories (e.g., ordered values, molecular fingerprints, model embeddings)?

See how to 
[encode](userguide/parameters.html#discrete-parameters)
discrete parameters or provide custom encodings.
```

## Account for Specifics of Data Availability or Data Acquisition Procedure

```{dropdown} Is no prior data available and the experiments should be done in batches?

Use [clustering](userguide/recommenders.html#clustering-recommenders) or [sampling](userguide/recommenders.html#sampling-recommenders) recommenders to diversify the first batch of parameter configurations to be tested.
```

```{dropdown} Is additional data from historic or other partially-related experiments available?

Use [transfer learning](userguide/transfer_learning.html).
```

```{dropdown} Will the outcome measurements of different parameter configurations become available at different times?

Use [asynchronous workflows](userguide/async.html).
```

## Advanced: Adjust how Recommendations are Prioritized

```{dropdown} Is the aim to reduce the overall uncertainty across different regions of the search space rather than optimize a specific objective?

Use [active learning](userguide/active_learning.html).
```
