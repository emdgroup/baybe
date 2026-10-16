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

## LLM-based recommenders

BayBE offers support for both pure and meta recommenders that query a Large Language
Model (LLM) for suggestions. LLMs can bring broad domain knowledge to the
optimization loop that is difficult to encode in traditional surrogate models — for
example, chemical intuition about which solvents are likely to dissolve a reactant,
or engineering knowledge about which parameter combinations are physically meaningful.

### Provider access via LiteLLM

BayBE uses [LiteLLM](https://docs.litellm.ai) as its model abstraction layer.
LiteLLM exposes a single `completion()` call that routes to Anthropic, OpenAI,
Azure, Google, and dozens of other providers without requiring provider-specific
code. The model is identified by a provider-prefixed string passed to the
`model` argument of {class}`~baybe.recommenders.pure.llm.llm.LLMRecommender`,
for example ``"anthropic/claude-sonnet-4-6"`` or ``"openai/gpt-4o"``.

Authentication is handled entirely through environment variables — BayBE does not
accept credentials as constructor arguments to prevent them from appearing in logs
or serialized campaign files. Set the variable that LiteLLM expects for your
provider before using an LLM recommender, e.g. ``ANTHROPIC_API_KEY`` for Anthropic
models or ``OPENAI_API_KEY`` for OpenAI. Refer to the
[LiteLLM provider documentation](https://docs.litellm.ai/docs/providers) for the
full list of supported providers and their environment variables.

Additional keyword arguments (e.g. ``temperature``, ``max_tokens``) can be forwarded
to LiteLLM via the `litellm_args` parameter.

### Prompt construction

On each call to
{meth}`~baybe.recommenders.pure.llm.llm.LLMRecommender.recommend`, BayBE
constructs a structured prompt containing the experiment description, the
serialized objective, parameters and constraints, a table of previous measurements,
and any pending or forbidden configurations. The model is asked to return a JSON
array of suggestions. If the response cannot be parsed or contains invalid values,
BayBE makes a single automatic recovery attempt by re-sending the full context
along with a description of what went wrong.

### Enriching recommendations with natural language

The quality of LLM suggestions improves when domain context is provided. BayBE
exposes `metadata` fields on parameters and objectives specifically for this
purpose. The `description` and `unit` keys have a defined meaning; any additional
keys are stored under `misc` and are also rendered into the prompt:

~~~python
from baybe.parameters import NumericalDiscreteParameter, SubstanceParameter

temperature = NumericalDiscreteParameter(
    "Temp_C",
    values=[90, 105, 120],
    metadata={"description": "Reaction temperature", "unit": "°C"},
)

solvent = SubstanceParameter(
    "Solvent",
    data={"DMAc": "CC(N(C)C)=O", "p-Xylene": "CC1=CC=C(C)C=C1"},
    encoding="MORDRED",
    metadata={
        "description": "Organic solvent",
        "role": "Controls polarity and solvation of reactants",
    },
)
~~~

The `experiment_description` string on the recommender itself is the most direct
lever: a concise but informative description of the experimental setup helps the
model reason about which parameter combinations are worth exploring.

### Available recommenders

* {class}`~baybe.recommenders.pure.llm.llm.LLMRecommender` is the pure recommender
  that queries the language model for every batch throughout the campaign. It is the
  building block for the two meta recommenders below and can also be used standalone:

  ~~~python
  from baybe.recommenders import LLMRecommender

  recommender = LLMRecommender(
      model="anthropic/claude-sonnet-4-6",
      experiment_description="Optimize a Suzuki coupling reaction.",
  )
  ~~~

* {class}`~baybe.recommenders.meta.llm.LLMTwoPhaseRecommender` is a specialization
  of {class}`~baybe.recommenders.meta.sequential.TwoPhaseMetaRecommender`
  whose initial recommender must be an
  {class}`~baybe.recommenders.pure.llm.llm.LLMRecommender`. The LLM proposes
  experiments until `switch_after` measurements have been collected, then a
  Bayesian recommender takes over permanently.

  ~~~python
  from baybe.recommenders import LLMTwoPhaseRecommender

  recommender = LLMTwoPhaseRecommender.from_model(
      model="anthropic/claude-sonnet-4-6",
      experiment_description="Optimize a Suzuki coupling reaction.",
      switch_after=10,
  )
  ~~~

* {class}`~baybe.recommenders.meta.llm.LLMAlternatingRecommender` is a specialization
  of {class}`~baybe.recommenders.meta.sequential.SequentialMetaRecommender`
  that cycles between an {class}`~baybe.recommenders.pure.llm.llm.LLMRecommender`
  and another recommender every time new measurements become available. This keeps
  the LLM's prior knowledge active throughout the campaign while Bayesian
  optimization continuously refines the surrogate model in alternating batches:

  ~~~python
  from baybe.recommenders import LLMAlternatingRecommender

  recommender = LLMAlternatingRecommender.from_model(
      model="anthropic/claude-sonnet-4-6",
      experiment_description="Optimize a Suzuki coupling reaction.",
  )
  ~~~

### Limitations

{class}`~baybe.recommenders.pure.llm.llm.LLMRecommender` does not support
{class}`~baybe.constraints.discrete.DiscreteCustomConstraint` or
{class}`~baybe.transformations.basic.CustomTransformation`, because their logic
is defined by arbitrary Python callables that cannot be communicated to the model.
Using either raises an
{class}`~baybe.exceptions.IncompatibilityError`.