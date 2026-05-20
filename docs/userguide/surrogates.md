# Surrogates

In Bayesian optimization (BO), surrogate models serve as a probabilistic stand-in for
the unknown system under study, modeling it from an input-output perspective. Based on
observations collected through experiments, surrogates provide predictions and
corresponding uncertainty estimates that guide us in deciding where to sample next.

## The Gaussian Process

By far the most widely used surrogate in Bayesian optimization – and thus unsurprisingly
BayBE's default choice – is the Gaussian process (GP). Its dominance stems from a
combination of properties that make it uniquely suited for the sequential
decision-making setting of experimental campaigns, making it the de facto workhorse
for most real-world applications:

- It provides a closed-form **joint posterior distribution** (mean and covariance) over
  the modeled system response for any set of candidate inputs. This information is
  key for balancing exploration/exploitation and batch-optimizing recommendations
  for parallel experimentation.
- It is **data-efficient**, i.e., with careful calibration, even a handful of
  observations can produce informative predictions and covariance structures, which is
  critical in BO-like settings where each experiment is expensive.
- It is **non-parametric**, meaning it can easily adapt to a wide range of function
  shapes without requiring manual specification of a fixed functional form. This is
  crucial for scientific discovery, i.e. when the underlying input-output relationships
  of the studied systems are complex and/or unknown.
- It offers many **mathematical conveniences**: closed-form expressions for
  gradients of the posterior (enabling efficient gradient-based optimization of
  acquisition functions), analytic formulas for marginal likelihoods (used for
  hyperparameter tuning), exact conditioning on observations, and straightforward
  computation of prediction intervals – all without requiring sampling or other
  approximation.

In BayBE, the GP surrogate is implemented in the
{class}`~baybe.surrogates.gaussian_process.core.GaussianProcessSurrogate`, which offers
complete configurability of its components:

### Customization

The behavior of BayBE's
{class}`~baybe.surrogates.gaussian_process.core.GaussianProcessSurrogate` is governed by
four configurable components:

| Component | Role |
|-----------|------|
| **Kernel** | Encodes assumptions about the function's structure. |
| **Mean function** | The expected function values prior to observing any data. |
| **Likelihood** | Encodes assumptions about the observation noise. |
| **Fit criterion** | The optimization objective used to tune the model hyperparameters. |

Configuring these components in the right way is key to unlocking the full potential of
the Bayesian optimization process since they directly drive the surrogate's predictions
and thus its ability to describe your data. Therefore, we invest much effort into
making BayBE's *default GP configuration* a solid, well-tested choice for a wide range
of problems, increasing the chances that you'll get decent performance out of the box.

However, we also understand that there is no one-size-fits-all solution and that certain
problems require custom modeling choices. To make this process as flexible and
user-friendly as possible, BayBE allows you to specify each of these components in
multiple ways:
* As a BayBE object representing the component (e.g., a
  {class}`~baybe.kernels.base.Kernel` object)
* As a low-level [GPyTorch](https://gpytorch.ai/) component (e.g., a
  {class}`gpytorch.kernels.Kernel`)
* Or most flexibly: as a *factory* producing either of the two above.
  More precisely: a callable that receives the actual recommendation context and
  dynamically returns a component produced specifically for the problem at hand. 

The following example demonstrates some of the possible specification mechanisms:
```python
import math

from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import LinearMean
from gpytorch.priors import GammaPrior
from torch import Tensor

from baybe.kernels.basic import LinearKernel, MaternKernel
from baybe.searchspace.core import SearchSpace
from baybe.surrogates import GaussianProcessSurrogate
from baybe.surrogates.gaussian_process.components import FitCriterion


def likelihood_factory(searchspace: SearchSpace, train_x: Tensor, train_y: Tensor):
    """Create a dimensionality-adjusted Gaussian likelihood model."""
    # Use a Gamma prior whose mean and variance scale with the input dimensionality
    d = train_x.shape[-1]
    noise_prior = GammaPrior(concentration=math.sqrt(d), rate=1.0)
    return GaussianLikelihood(noise_prior)


surrogate = GaussianProcessSurrogate(
    kernel_or_factory=5 * MaternKernel() + LinearKernel(),  # BayBE kernel arithmetic
    mean_or_factory=LinearMean(input_size=3),  # GPyTorch mean
    likelihood_or_factory=likelihood_factory,  # Factory producing GPyTorch likelihood
    fit_criterion_or_factory=FitCriterion.MARGINAL_LOG_LIKELIHOOD,  # Enum value
)
```

For more details, have a look at the
{class}`~baybe.surrogates.gaussian_process.core.GaussianProcessSurrogate` documentation.

#### Kernels

Selecting a {class}`~baybe.kernels.base.Kernel` is arguably the most impactful modeling
choice in a GP. It encodes prior beliefs about the target function – for instance, how
smooth it is, whether it exhibits periodic patterns, or how quickly correlations decay
with distance.

Due to their central role in GP modeling, kernels are treated as first-class citizens in
BayBE (divided into [basic](baybe.kernels.basic) and
[composite](baybe.kernels.composite) kernels), providing high-level access to features
that facilitate the modeling process:
* Kernels can be conveniently composed using basic arithmetic operations to express rich
  structural assumptions in a modular way.
* Kernels can be restricted to act on specific parameters, enabling model architectures
  where different kernels focus on different dimensions of the input space.

The following example gives a taste of what kernel composition could look like for your
use case. For more inspiration, have a look at the [kernel
cookbook](https://www.cs.toronto.edu/~duvenaud/cookbook/).
```python
from baybe.kernels.basic import MaternKernel, PeriodicKernel
from baybe.kernels.composite import AdditiveKernel, ProductKernel, ScaleKernel
from baybe.priors import GammaPrior

# Kernel addition
kernel = MaternKernel() + PeriodicKernel()
kernel = AdditiveKernel([MaternKernel(), PeriodicKernel()])  # equivalent explicit form

# Kernel multiplication
kernel = MaternKernel() * PeriodicKernel()
kernel = ProductKernel([MaternKernel(), PeriodicKernel()])  # equivalent explicit form

# Scaling a kernel by a *fixed* constant vs. a *learnable* constant
kernel = 2.0 * MaternKernel()
kernel = ScaleKernel(
    MaternKernel(),
    outputscale_prior=GammaPrior(2.0, 0.15),
    outputscale_initial_value=2.0,
)

# Restricting a kernel to specific parameters
kernel = MaternKernel(parameter_names=["Param_A", "Param_B"])
```

#### Presets

Providing customized GP configurations on the component level is powerful for expert
users who want to fine-tune every aspect of their surrogate model. However, it can be
daunting for newcomers or those who lack the background to make informed decisions about
individual GP components.

For users who want well-tested GP configurations without manually specifying each
component, BayBE offers *presets* – curated bundles of GP components emulating
configurations from the literature or other software packages that can be activated with
a single line of code:

```python
from baybe.surrogates import GaussianProcessSurrogate
from baybe.surrogates.gaussian_process.presets import GaussianProcessPreset

surrogate = GaussianProcessSurrogate.from_preset(GaussianProcessPreset.CHEN)
surrogate = GaussianProcessSurrogate.from_preset("CHEN")  # equivalent string access
```

See [here](baybe.surrogates.gaussian_process.presets.core.GaussianProcessPreset) for the
full list of available presets. To understand their respective purpose and effects, we
recommend reading the mentioned sources.


````{admonition} Customizing Presets
:class: info

Presets can also serve as a **starting point** for further customization, i.e., any
preset component can be individually overridden:

```python
# Use the "CHEN" preset but swap in a custom kernel
surrogate = GaussianProcessSurrogate.from_preset(
    GaussianProcessPreset.CHEN,
    kernel_or_factory=MaternKernel(nu=1.5, parameter_names=["Param_A", "Param_B"]),
)
```
````


## Alternative Surrogates

Unless you have a very specific reason to choose otherwise, the Gaussian process should
likely be considered your first-choice surrogate, for the [reasons explained
above](#the-gaussian-process). However, there are scenarios where it may be necessary to
consider other options, for example, when dealing with large datasets and computational
cost becomes a major concern. For these cases, BayBE offers a range of alternative
surrogate models:


| Surrogate | Description |
|-----------|-------------|
| {class}`~baybe.surrogates.linear.BayesianLinearSurrogate` | A lightweight surrogate with built-in uncertainty quantification. Suitable when the underlying relationship is approximately linear in the given features. Complexity scales with the number of features, not with the number of data points, making it an attractive alternative to the GP for large datasets. |
| {class}`~baybe.surrogates.random_forest.RandomForestSurrogate` | Tree-based surrogate. Scales well to large datasets and can capture non-smooth relationships. Uncertainty is estimated from the tree ensemble. |
| {class}`~baybe.surrogates.ngboost.NGBoostSurrogate` | Gradient boosting with natural gradients for probabilistic predictions. |


## Custom Surrogates

Next to our built-in surrogate models, BayBE welcomes you to bring your own custom model
for advanced workflows. There are at least two reasons why you might want to consider
this:
* It allows you to leverage existing models that you have already developed and validated
  outside of the BayBE ecosystem.
* For [*grey box modeling*](https://en.wikipedia.org/wiki/Grey_box_model): you have
  domain-specific knowledge (e.g. physical constraints, expert insights) that you can
  inject into the surrogate, which can drastically speed up the optimization process.

BayBE offers two convenient mechanisms to integrate your models:

### The `SurrogateProtocol`
The BayBE ecosystem relies on a lightweight interface for surrogate models – the
{class}`~baybe.surrogates.base.SurrogateProtocol` – which allows you to easily plug in
your own objects. As long as your model obeys this protocol, you can use it in any
context where otherwise built-in surrogates would be called.

To enable it, your object must essentially expose two simple mechanisms:
* A {meth}`~baybe.surrogates.base.SurrogateProtocol.fit` method, which specifies how
  the surrogate is to be trained for a given recommendation context.
* A {meth}`~baybe.surrogates.base.SurrogateProtocol.to_botorch` method, which defines
  how the trained surrogate can be turned into a BoTorch-compatible
  {class}`~botorch.models.model.Model` object.


### ONNX Surrogates
In addition to the protocol, you can use pretrained surrogates in the [ONNX
format](https://onnx.ai/) via our {class}`~baybe.surrogates.custom.CustomONNXSurrogate` class.
To see a concrete workflow of how to use this surrogate type, have a look at [this
example](./../../examples/Custom_Surrogates/custom_pretrained.py).


(multi_output_modeling)= 
## Multi-Output Modeling

When modeling multiple output variables simultaneously, not all surrogate types
natively provide (joint) predictive distributions for more than one variable, as
indicated by their
{attr}`~baybe.surrogates.base.Surrogate.supports_multi_output` attribute.

In multi-output contexts, it may therefore be necessary to assemble several
single-output surrogates into a composite model to build a joint predictive model from
independent components for each output. BayBE provides two convenient mechanisms to
achieve this, both built upon the
{class}`~baybe.surrogates.composite.CompositeSurrogate` class:

### Surrogate Replication
The simplest way to construct a multi-output surrogate is to replicate a given
single-output model architecture for each of the existing output dimensions.

To replicate a given surrogate, you can either call its 
{meth}`~baybe.surrogates.base.Surrogate.replicate` method or use the
[`CompositeSurrogate.from_replication()`](baybe.surrogates.composite.CompositeSurrogate.from_replication)
convenience constructor:
```python
from baybe.surrogates import CompositeSurrogate, GaussianProcessSurrogate

composite_a = GaussianProcessSurrogate().replicate()
composite_b = CompositeSurrogate.from_replication(GaussianProcessSurrogate())

assert composite_a == composite_b
```

However, there are very few cases where such an explicit conversion is required. Because
using a single-output surrogate model in a multi-output context would trivially fail, and
because BayBE cares deeply about its users' lives, it automatically performs this conversion
for you behind the scenes:

(auto_replication)=
```{admonition} Auto-Replication
:class: important

When using a single-output surrogate model in a multi-output context, BayBE
automatically replicates the surrogate on the fly.
```
The consequence of the above is that you can use the same model object regardless
of the modeling context and its multi-output capabilities.

There is *one* notable exception where an explicit replication may still make
sense: if you want to bypass the existing multi-output mechanics of a surrogate that is
inherently multi-output compatible.

### Composite Surrogates
An alternative to surrogate replication is to manually assemble your
{class}`~baybe.surrogates.composite.CompositeSurrogate`. This can be useful if you want
to
* use the same model architecture but with different settings for each output or
* use different architectures for the outputs to begin with.

```python
from baybe.surrogates import (
    CompositeSurrogate,
    GaussianProcessSurrogate,
    RandomForestSurrogate,
)

surrogate = CompositeSurrogate(
    {
        "target_a": GaussianProcessSurrogate(),
        "target_b": RandomForestSurrogate(),
    }
)
```

A noticeable difference to the replication approach is that manual assembly requires
the exact set of target variables to be known at the time the object is created.



## Extracting the Model for Advanced Study

Because the surrogate model is automatically retrained as soon as new data becomes
available during the Bayesian optimization process, it is a short-lived object that does
not require persistent storage. However, it can still be useful to access the surrogate
model for advanced study, such as investigating the posterior predictions, acquisition
functions, or feature importance.

For this purpose, BayBE provides the ``get_surrogate`` method, which is available for
the [``Campaign``](baybe.campaign.Campaign.get_surrogate) or for
[recommenders](baybe.recommenders.pure.bayesian.base.BayesianRecommender.get_surrogate),
and gives you direct access to the most up-to-date model.

Below an example of how to use this functionality to extract the posterior mean of the model:

~~~python
# Assuming we already have a campaign created and measurements added
data = campaign.measurements[[p.name for p in campaign.parameters]]
posterior_mean = lambda x: campaign.get_surrogate().posterior(x).mean
~~~

```{admonition} Insights Module
:class: info
Note that BayBE has a dedicated [Insights module](./insights.md) that offers certain
analytical tools and visualizations out-of-the-box, without requiring manual extraction
of the surrogate model to begin with.
```

