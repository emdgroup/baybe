# Recommenders

## General information

Recommenders are an essential part of BayBE that effectively explore the search space and provide recommendations for the next experiment or batch of experiments. While some recommenders are versatile and work across different types of search spaces, other are specifically designed for discrete or continuous spaces. The compatibility is indicated via the corresponding ``compatibility`` class variable.

The set of available recommenders can be partitioned into the following subclasses.

## Bayesian recommenders

The Bayesian recommenders in BayBE are built on the foundation of the [`BayesianRecommender`](baybe.recommenders.bayesian.BayesianRecommender) class, offering an array of possibilities with internal surrogate models and support for various acquisition functions.

The [`SequentialGreedyRecommender`](baybe.recommenders.bayesian.SequentialGreedyRecommender) is a powerful recommender that leverages BoTorch optimization functions to perform sequential Greedy optimization. It can be applied for discrete, continuous and hybrid sarch spaces. It is an implementation of the BoTorch optimization functions for discrete, continuous and mixed spaces. 

It is important to note that this recommender performs a brute-force search when applied in hybrid search spaces, as it optimizes the continuous part of the space while exhaustively searching choices in the discrete subspace. You can customize this behavior to only sample a certain percentage of the discrete subspace via the ``sample_percentage`` attribute and to choose different sampling strategies via the ``hybrid_sampler`` attribute. An example on using this recommender in a hybrid space can be found [here](./../../examples/Backtesting/hybrid).

The [`NaiveHybridRecommender`](baybe.recommenders.bayesian.NaiveHybridRecommender) can be applied to all search spaces, but is intended to be used in hybrid spaces. This recommender combines individual recommenders for the continuous and the discrete subspaces. It independently optimizes each subspace and consolidates the best results to generate a candidate for the original hybrid space. An example on using this recommender in a hybrid space can be found [here](./../../examples/Backtesting/hybrid).

## Clustering recommenders

BayBE offers a set of recommenders leveraging clustering techniques to facilitate initial point selection:
* **[`PAMClusteringRecommender`](baybe.recommenders.clustering.PAMClusteringRecommender):** This recommender utilizes partitioning around medoids for effective clustering.
* **[`KMeansClusteringRecommender`](baybe.recommenders.clustering.KMeansClusteringRecommender):** This recommender implements the k-means clustering strategy.
* **[`GaussianMixtureClusteringRecommender`](baybe.recommenders.clustering.GaussianMixtureClusteringRecommender):** This recommender leverages Gaussian Mixture Models for clustering.

## Sampling recommenders

BayBE provides two sampling-based recommenders:

* **[`RandomRecommender`](baybe.recommenders.sampling.RandomRecommender):** This recommender offers random recommendations for all types of search spaces. This recommender is extensively used in backtesting examples, providing a valuable comparison. For detailed usage examples, refer to the examples listed [here](./../../examples/Backtesting/Backtesting).
* **[`FPSRecommender`](baybe.recommenders.sampling.FPSRecommender):** This recommender is only applicable for discrete search spaces, and recommends points based on farthest point sampling. A practical application showcasing the usage of this recommender can be found [here](./../../examples/Custom_Surrogates/surrogate_params).