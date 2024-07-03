# Transfer Learning

BayBE offers the possibility to mix data from multiple, *similar but not identical*
campaigns in order to accelerate optimization. 
Using data from multiple campaigns is currently supported when using a Gaussian Process
Surrogate model and is implemented by the [`TaskParameter`](baybe.parameters.categorical.TaskParameter).

```{admonition} Terminology
:class: note
The term "Transfer Learning" is used in a lot of different ways. Within BayBE,
"Transfer Learning" refers to combining data from multiple campaigns that are from 
similar contexts. Depending on the field, this might also be known as "contextual
learning". In BayBE we also refer to the different contexts as "tasks".
```

## When and When Not To Use Transfer Learning

Thinking about combining data from different campaigns, a naive approach to achieve
this is to quantify the differences between the contexts via one or few explicitly
measured parameters. Examples where this is possible are:
* Data for a chemical reaction that was optimized at a certain temperature can be used 
  in a new campaign, where the same reaction should be optimized again at a different 
  temperature.
* Optimization of a simulation involving different particle sizes: The data obtained at
  a smaller particle size can be utilized when starting a new optimization for a larger
  particle size.

However, it can be difficult to quantify the differences between the campaigns. This
might be the case if the parameters distinguishing the contexts i) are not available and
cannot be measured anymore; ii) are too many and explicitly modelling them is out of
question; or iii) are simply unknown.

Examples for situations where explicit quantification of the context can be difficult:
* Cell culture media optimizations for different cell types: Cell types differ among 
  many possible descriptors, and it is not known a priori which ones are relevant to a
  newly started campaign.
* Optimization in industrial black-box contexts: When materials (such as cell lines or
  complex substances) stem from customers, they can come uncharacterized.
* Transfer of a complicated process to another location. The transferred machinery will
  likely require a new calibration / optimization and could benefit from the other
  location's data. However, is not necessarily clear what parameters differentiate the
  location context.

**Transfer Learning** in BayBE offers something for examples such as the latter, because
it abstracts the differences between campaigns into one dimension. Over the course of
an ongoing campaign, the covariance between current campaign data and data from
previous campaigns can be learned, basically enabling us to utilize previous data. For
instance, this can unlock data treasures coming from similar but not identical campaigns
accumulated over many years.

```{admonition} Expectations
:class: important
Because of the need to learn the covariance between tasks, transfer learning is not a
magic method for zero-shot learning. It will always need data from an ongoing campaign
to understand how other campaigns' data are related. Overall, if correlated task data
were provided, the optimization of new campaigns can experience a dramatic speedup.
```

## The role of `TaskParameter`

The `TaskParameter` is used to "mark" the context of an individual experiment. The
set of all possible contexts is provided upon the initialization of a `TaskParameter`
by providing them as `values`.
In the following example, the context might be one of several reactors in which
a chemical experiments can be conducted.

```python
from baybe.parameters import TaskParameter

TaskParameter(name="Reactor", values=["ReactorA", "ReactorB", "ReactorC"])
```

If not specified further, a campaign using the `TaskParameter` as specified above
would now make recommendations for all possible values of the parameter. Using the
`active_values` argument upon initialization, this behavior can be changed such that
the `campaign` only makes recommendations for the corresponding values.

The following example models a situation in which data from experiments in three
different reactors are available, but new experiments should only be conducted in
`ReactorC`.

```python
from baybe.parameters import TaskParameter

TaskParameter(
    name="Reactor",
    values=["ReactorA", "ReactorB", "ReactorC"],
    active_values=["ReactorC"],
)
```

This can be abstracted easily to other scenarios such as changing substrates (while
screening same reaction conditions) or formulating mixtures for different cell lines:

~~~python
TaskParameter(
    name="Substrate",
    values=["3,5-dimethylisoxazole", "benzo[d]isoxazole", "5-methylisoxazole"],
    active_values=["3,5-dimethylisoxazole"],
)
TaskParameter(
    name="Cell_Line",
    values=["Liver cell", "Heart cell", "Hamster brain cell"],
    active_values=["Liver cell"],
)
~~~

```{admonition} Technology
:class: note
Note that the act of combining the data of several contexts can be done via
multiple models, shared architectures, special kernels in a single model, etc. We do
not necessarily want to limit to any of these methods, even though BayBE currently
only implements a variant via kernels.
```

## Seeing Transfer Learning in Action

We provide full example demonstrating BayBE's transfer learning capabilities.
We want to briefly discuss and highlight the results of this example in this user guide.
The full example can be found [here](../../examples/Transfer_Learning/basic_transfer_learning).

The example optimizes an analytical function, the so-called "Hartmann Function".
We use transfer learning by providing additional data that was obtained by evaluating a
negated noisy variant of the same function.

The following plot demonstrates the effect that providing this additional data has:

```{image} ../../examples/Transfer_Learning/basic_transfer_learning_light.svg
:align: center
:class: only-light
```

```{image} ../../examples/Transfer_Learning/basic_transfer_learning_dark.svg
:align: center
:class: only-dark
```