# Transfer Learning

BayBE offers the possibility to mix data from multiple campaigns in order to accelerate
optimization. 
Using data from multiple campaigns is currently supported when using a Gaussian Process
Surrogate model and is implemented by the [`TaskParameter`](baybe.parameters.categorical.TaskParameter).

```{admonition} Terminology
:class: note
The term "Transfer Learning" is used in a lot of different ways.
Within BayBE, "Transfer Learning" refers to combining data from multiple campaigns.
Depending on the field, this might also be known as "contextual learning".
Further, note that the act of combining the data of several contexts can be done via
multiple models, shared architectures, special kernels in a single model, etc.
We do not necessarily want to limit to any of these methods, even though BayBE currently
offers only a single one.
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
    name="Month",
    values=["Liver cell", "Heart cell", "Hamster brain cell"],
    active_values=["Liver cell"],
)
~~~

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