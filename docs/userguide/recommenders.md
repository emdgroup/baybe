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

```{admonition} Additional Options for Discrete Search Spaces
:class: note
For discrete search spaces, BayBE provides additional control over pure recommenders.
You can explicitly define whether a recommender is allowed to recommend previous 
recommendations again via `allow_repeated_recommendations` and whether it can output
recommendations that have already been measured via
`allow_recommending_already_measured`. 
```

### Bayesian Recommenders

The Bayesian recommenders in BayBE are built on the foundation of the
[`BayesianRecommender`](baybe.recommenders.pure.bayesian.base.BayesianRecommender)
class, offering an array of possibilities with internal surrogate models and support
for various acquisition functions.

* The **[`SequentialGreedyRecommender`](baybe.recommenders.pure.bayesian.sequential_greedy.SequentialGreedyRecommender)**
  is a powerful recommender that performs sequential Greedy optimization. It can be
  applied for discrete, continuous and hybrid search spaces. It is an implementation of
  the BoTorch optimization functions for discrete, continuous and mixed spaces.
  It is important to note that this recommender performs a brute-force search when
  applied in hybrid search spaces, as it optimizes the continuous part of the space
  while exhaustively searching choices in the discrete subspace. You can customize
  this behavior to only sample a certain percentage of the discrete subspace via the
  `sample_percentage` attribute and to choose different sampling algorithms via the
  `hybrid_sampler` attribute. An example on using this recommender in a hybrid space
  can be found [here](./../../examples/Backtesting/hybrid).

* The **[`NaiveHybridSpaceRecommender`](baybe.recommenders.naive.NaiveHybridSpaceRecommender)**
  can be applied to all search spaces, but is intended to be used in hybrid spaces.
  This recommender combines individual recommenders for the continuous and the discrete
  subspaces. It independently optimizes each subspace and consolidates the best results
  to generate a candidate for the original hybrid space. An example on using this
  recommender in a hybrid space can be found [here](./../../examples/Backtesting/hybrid).

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
    TwoPhaseMetaRecommender,
    RandomRecommender,
    SequentialGreedyRecommender,
)

recommender = TwoPhaseMetaRecommender(
    initial_recommender=RandomRecommender(), recommender=SequentialGreedyRecommender()
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
  Due to the arbitrary nature of iterables that can be used, de-/serializability cannot
  be guaranteed. As a consequence, using a `StreamingSequentialMetaRecommender` results
  in an error if you attempt to serialize the corresponding object or higher-level
  objects containing it.
  ```