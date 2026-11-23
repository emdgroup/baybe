# Recommenders

## General Information

Recommenders are an essential part of BayBE that effectively explore the search space
and provide recommendations for the next experiment or batch of experiments.
Available recommenders can be partitioned into the following subclasses.

## Pure Recommenders

Pure recommenders simply take on the task to recommend measurements. They each contain
the inner logic to do so via different algorithms and approaches.
While some pure recommenders are versatile and work across different types of search
spaces, other are specifically designed for discrete or continuous spaces. The
compatibility is indicated via the corresponding `compatibility` class variable.

### Bayesian Recommenders

The Bayesian recommenders in BayBE are built on the foundation of the
[`BayesianRecommender`](baybe.recommenders.pure.bayesian.base.BayesianRecommender)
class, offering an array of possibilities with internal surrogate models and support
for various acquisition functions.

* The **[`BotorchRecommender`](baybe.recommenders.pure.bayesian.botorch.core.BotorchRecommender)**
  is a powerful recommender based on BoTorch's optimization engine that can be applied
  to all kinds of search spaces. In continuous spaces, its `sequential_continuous` flag
  allows to choose between greedy sequential optimization and batch optimization as the
  underlying point generation mode. In discrete/hybrid spaces, sequential greedy
  selection is the only available mode and is thus activated automatically.
  
  Note that the recommender performs a brute-force search when applied to hybrid search
  spaces, as it does gradient-based optimization in the continuous part of the space
  while exhaustively evaluating configurations of the discrete subspace. You can customize this
  behavior to only sample a certain percentage of the discrete subspace via the
  {attr}`~baybe.recommenders.pure.bayesian.botorch.core.BotorchRecommender.sampling_percentage`
  argument and to choose different sampling algorithms via the
  {attr}`~baybe.recommenders.pure.bayesian.botorch.core.BotorchRecommender.hybrid_sampler`
  argument.

  The gradient-based optimization part can also further be controlled by the
  {attr}`~baybe.recommenders.pure.bayesian.botorch.core.BotorchRecommender.n_restarts` and
  {attr}`~baybe.recommenders.pure.bayesian.botorch.core.BotorchRecommender.n_raw_samples`
  arguments. For details, please refer
  to [BotorchRecommender](baybe.recommenders.pure.bayesian.botorch.core.BotorchRecommender).
 
* The **[`NaiveHybridSpaceRecommender`](baybe.recommenders.naive.NaiveHybridSpaceRecommender)**
  can be applied to all search spaces, but is intended to be used in hybrid spaces.
  This recommender combines individual recommenders for the continuous and the discrete
  subspaces. It independently optimizes each subspace and consolidates the best results
  to generate a candidate for the original hybrid space. 

(recommender_data_augmentation)=
#### Data Augmentation

In certain situations like [mixture modeling](/examples/Mixtures/slot_based),
symmetries are present. Data augmentation is a model-agnostic way of enabling the
surrogate model to learn such symmetries effectively, which might result in a better
performance, similar as e.g. for image classification models. BayBE automatically
performs data augmentation when symmetries are assigned to the
{attr}`~baybe.recommenders.pure.bayesian.base.BayesianRecommender.symmetries`
attribute of the recommender. This means you can add a data point in any acceptable
representation and BayBE will train the model on this point plus augmented points that
can be generated from it. To see the effect in practice, refer to
[this example](/examples/Symmetries/permutation).

### Clustering Recommenders

BayBE offers a set of recommenders leveraging techniques to facilitate point selection
via clustering:
* **[`PAMClusteringRecommender`](baybe.recommenders.pure.nonpredictive.clustering.PAMClusteringRecommender):**
  This recommender utilizes partitioning around medoids.
* **[`KMeansClusteringRecommender`](baybe.recommenders.pure.nonpredictive.clustering.KMeansClusteringRecommender):**
  This recommender implements k-means clustering.
* **[`GaussianMixtureClusteringRecommender`](baybe.recommenders.pure.nonpredictive.clustering.GaussianMixtureClusteringRecommender):**
  This recommender leverages Gaussian Mixture Models for clustering.

### Sampling Recommenders

BayBE provides two recommenders that recommend by sampling form the search space:
* **[`RandomRecommender`](baybe.recommenders.pure.nonpredictive.sampling.RandomRecommender):**
  This recommender offers random recommendations for all types of search spaces.
  It is extensively used in backtesting examples, providing a valuable comparison.
  For detailed usage examples, refer to the list
  [here](./../../examples/Backtesting/Backtesting).
* **[`FPSRecommender`](baybe.recommenders.pure.nonpredictive.sampling.FPSRecommender):**
  This recommender is only applicable for discrete search spaces, and recommends points
  based on farthest point sampling. A practical application showcasing the usage of
  this recommender can be found
  [here](./../../examples/Custom_Surrogates/surrogate_params).

(llm_recommender)=
### LLM Recommender

The {class}`~baybe.recommenders.pure.llm.llm.LLMRecommender` asks a large language
model (LLM) for recommendations. It brings in domain knowledge that is not encoded in
a surrogate model, which is useful when little or no data is available.
Note that it is not Bayesian: there is no surrogate model and no uncertainty estimate.

#### Installation and Model Access

The recommender is available the ``llm`` extra and sends requests through
[LiteLLM](https://docs.litellm.ai). The ``model`` argument is a provider-prefixed
LiteLLM identifier; see the
[list of supported providers](https://docs.litellm.ai/docs/providers). Credentials are
taken exclusively from the environment variables LiteLLM expects for the provider
(e.g. ``OPENAI_API_KEY`` or ``ANTHROPIC_API_KEY``). BayBE deliberately offers no
credential argument so that keys cannot end up in serialized campaigns.

Additional LiteLLM arguments such as ``temperature`` or ``max_tokens`` are passed via
``litellm_args``. Their values are neither shown in the representation nor serialized,
so they must be provided again after deserialization (a warning is raised when they are
dropped). The keys ``model``, ``messages`` and ``stream`` are controlled by BayBE and
rejected.

```python
from baybe.recommenders import LLMRecommender

recommender = LLMRecommender(
    model="openai/gpt-4o-mini",
    experiment_description=(
        "Suzuki coupling of an aryl bromide with a boronic acid in a batch "
        "reactor. The catalyst loading is fixed."
    ),
    litellm_args={"temperature": 0.2},
)
```

The recommender is then used like any other recommender, e.g. inside a
{class}`~baybe.campaign.Campaign`. Constructing it does not contact the model; only
requesting recommendations does.

#### Information Passed to the Model

The prompt contains the requested batch size, the ``experiment_description``, the
serialized objective (including target metadata and descriptions of the used
transformations), the serialized parameters and constraints (including their metadata),
three example configurations sampled from the search space, the previous measurements,
and the pending experiments. For label-like parameters only the active values are sent;
for {class}`~baybe.parameters.substance.SubstanceParameter` these come with the SMILES
of the active labels, while the descriptor table of a
{class}`~baybe.parameters.custom.CustomDiscreteParameter` is omitted.

The two levers for better recommendations are a precise ``experiment_description`` and
the ``metadata`` of parameters, targets and the objective (see the
{ref}`parameter metadata <parameter_metadata>` section). If the objective has no
description, a ``UserWarning`` is raised.

```{admonition} Data Privacy
:class: warning
Everything listed above, including measurements, molecular structures and free-text
descriptions, is sent to the provider configured via ``model``. Check that this
complies with your data-handling policies before use.
```

#### Response Handling

The model must return exactly the requested number of distinct recommendations inside
the search space. Numerical discrete values are snapped to the nearest allowed value.
If a response is unusable, exactly one recovery attempt is made with the full context
and a description of the problem; if that fails too,
{class}`~baybe.exceptions.LLMRecoveryFailedError` is raised. Failures of the call
itself are not retried: {class}`~baybe.exceptions.LLMAuthenticationError` is raised if
the provider rejects authentication, and {class}`~baybe.exceptions.LLMCallError` for
everything else (network, rate limit, unknown model, missing credentials).

```{admonition} Limitations
:class: warning
Components whose behavior is defined by arbitrary Python callables cannot be described
to the model, which raises an {class}`~baybe.exceptions.IncompatibilityError`; examples
are {class}`~baybe.transformations.basic.CustomTransformation` and
{class}`~baybe.constraints.discrete.DiscreteCustomConstraint`.

Constraints of the continuous subspace cannot be validated. A
{class}`~baybe.exceptions.LLMResponseWarning` is raised and compliance is not
guaranteed.
```

## Meta Recommenders

In analogy to meta studies, meta recommenders are wrappers that operate on a sequence
of pure recommenders and determine when to switch between them according to different
logics. BayBE offers three distinct kinds of meta recommenders.

* The
  [`TwoPhaseMetaRecommender`](baybe.recommenders.meta.sequential.TwoPhaseMetaRecommender)
  employs two distinct recommenders and switches between them at a certain specified
  point, controlled by the `switch_after` attribute. This is useful e.g. if you want a
  different recommender for the initial recommendation when there is no data yet
  available. This simple example would recommend randomly for the first batch and switch
  to a Bayesian recommender as soon as measurements have been ingested:
```python
from baybe.recommenders import (
    BotorchRecommender,
    TwoPhaseMetaRecommender,
    RandomRecommender,
)

recommender = TwoPhaseMetaRecommender(
    initial_recommender=RandomRecommender(), recommender=BotorchRecommender()
)
```

* The **[`SequentialMetaRecommender`](baybe.recommenders.meta.sequential.SequentialMetaRecommender)**
  introduces a simple yet versatile approach by utilizing a predefined list of
  recommenders. By specifying the desired behavior using the `mode` attribute, it is
  possible to flexibly determine the meta recommender's response when it exhausts the 
  available recommenders. The possible choices are to either raise an error, re-use the
  last recommender or re-start at the beginning of the sequence.

* Similar to the `SequentialMetaRecommender`, the
  **[`StreamingSequentialMetaRecommender`](baybe.recommenders.meta.sequential.StreamingSequentialMetaRecommender)**
  enables the utilization of *arbitrary* iterables to select recommender.

  ```{warning}
  Due to the arbitrary nature of iterables that can be used, (de-)serializability cannot
  be guaranteed. As a consequence, using a `StreamingSequentialMetaRecommender` results
  in an error if you attempt to serialize the corresponding object or higher-level
  objects containing it.
  ```