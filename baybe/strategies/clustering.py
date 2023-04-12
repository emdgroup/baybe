# pylint: disable=missing-class-docstring, missing-function-docstring
# TODO: add docstrings

"""Recommendation strategies based on clustering."""

from abc import ABC
from typing import List, Optional, Type, TypeVar

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture

from sklearn.preprocessing import StandardScaler

from baybe.searchspace import SearchSpace, SearchSpaceType
from baybe.strategies.recommender import NonPredictiveRecommender

try:
    from sklearn_extra.cluster import KMedoids
except ImportError:
    KMedoids = None


SklearnModel = TypeVar("SklearnModel")


class SKLearnClusteringRecommender(NonPredictiveRecommender, ABC):
    """
    Intermediate class for cluster-based selection of discrete candidates.

    Suitable for sklearn-like models that have a `fit` and `predict` method. Specific
    model parameters and cluster sub-selection techniques can be declared in the
    derived classes.
    """

    type = "ABSTRACT_SKLEARN_CLUSTERING"
    compatibility = SearchSpaceType.DISCRETE

    # Properties that need to be defined by derived classes
    model_class: Type[SklearnModel]
    model_cluster_num_parameter_name: str

    def __init__(self, **kwargs):
        super().__init__()
        self.model_params = kwargs
        self._use_custom_selector = False

        # Members that will be initialized during the recommendation process
        self.model: Optional[SklearnModel] = None
        self.candidates_scaled: Optional[pd.DataFrame] = None
        self.scaler: Optional[StandardScaler] = None

    def _make_selection_default(self) -> List[int]:
        """
        Basic model-agnostic method that selects one candidate from each cluster
        uniformly at random.

        Returns
        -------
        selection : List[int]
           Positional indices of the selected candidates.
        """
        assigned_clusters = self.model.predict(self.candidates_scaled)
        selection = [
            np.random.choice(np.argwhere(cluster == assigned_clusters).flatten())
            for cluster in np.unique(assigned_clusters)
        ]
        return selection

    def _make_selection_custom(self) -> List[int]:
        """
        A model-specific method to select candidates from the computed clustering.
        May be implemented by the derived class.
        """
        raise NotImplementedError("This line in the code should be unreachable. Sry.")

    def _recommend_discrete(
        self,
        searchspace: SearchSpace,
        candidates_comp: pd.DataFrame,
        batch_quantity: int,
    ) -> pd.Index:
        """See base class."""
        # Fit scaler on entire searchspace
        # TODO [Scaling]: scaling should be handled by searchspace object
        self.scaler = StandardScaler()
        self.scaler.fit(searchspace.discrete.comp_rep)

        self.candidates_scaled = np.ascontiguousarray(
            self.scaler.transform(candidates_comp)
        )

        # Set model parameters and perform fit
        self.model_params.update(
            {self.model_cluster_num_parameter_name: batch_quantity}
        )
        self.model = self.model_class(**self.model_params)
        self.model.fit(self.candidates_scaled)

        # Perform selection based on assigned clusters
        if self._use_custom_selector:
            selection = self._make_selection_custom()
        else:
            selection = self._make_selection_default()

        # Convert positional indices into DataFrame indices and return result
        return candidates_comp.index[selection]


if KMedoids:
    # TODO: Instead of hiding the class, raise an error when attempting to create the
    #   object. However, this requires to replace the current class-based handling of
    #   recommenders with an object-based logic, since otherwise the exception can be
    #   triggered arbitrarily late in the DOE process.

    class PAMClusteringRecommender(SKLearnClusteringRecommender):
        """Partitioning Around Medoids (PAM) initial clustering strategy."""

        type = "CLUSTERING_PAM"
        model_class = KMedoids
        model_cluster_num_parameter_name = "n_clusters"

        def __init__(
            self, use_custom_selector: bool = True, max_iter: int = 100, **kwargs
        ):
            super().__init__(max_iter=max_iter, init="k-medoids++", **kwargs)
            self._use_custom_selector = use_custom_selector

        def _make_selection_custom(self) -> List[int]:
            """
            In PAM, cluster centers (medoids) correspond to actual data points,
            which means they can be directly used for the selection.
            """
            selection = self.model.medoid_indices_.tolist()
            return selection


class KMeansClusteringRecommender(SKLearnClusteringRecommender):
    """K-means initial clustering strategy."""

    type = "CLUSTERING_KMEANS"
    model_class = KMeans
    model_cluster_num_parameter_name = "n_clusters"

    def __init__(
        self,
        use_custom_selector: bool = True,
        n_init: int = 50,
        max_iter: int = 1000,
        **kwargs,
    ):
        super().__init__(n_init=n_init, max_iter=max_iter, **kwargs)
        self._use_custom_selector = use_custom_selector

    def _make_selection_custom(self) -> List[int]:
        """
        For K-means, a reasonable choice is to pick the points closest to each
        cluster center.
        """
        distances = pairwise_distances(
            self.candidates_scaled, self.model.cluster_centers_
        )
        # Set the distances of points that were not assigned by the model to that
        # cluster to infinity. This assures that one unique point per cluster is
        # assigned.
        predicted_clusters = self.model.predict(self.candidates_scaled)
        for k_cluster in range(self.model.cluster_centers_.shape[0]):
            idxs = predicted_clusters != k_cluster
            distances[idxs, k_cluster] = np.inf
        selection = np.argmin(distances, axis=0).tolist()
        return selection


class GaussianMixtureClusteringRecommender(SKLearnClusteringRecommender):
    """Gaussian mixture model (GMM) initial clustering strategy."""

    type = "CLUSTERING_GMM"
    model_class = GaussianMixture
    model_cluster_num_parameter_name = "n_components"

    def __init__(self, use_custom_selector: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._use_custom_selector = use_custom_selector

    def _make_selection_custom(self) -> List[int]:
        """
        In a GMM, a reasonable choice is to pick the point with the highest
        probability densities for each cluster.
        """
        predicted_clusters = self.model.predict(self.candidates_scaled)
        selection = []
        for k_cluster in range(self.model.n_components):
            density = multivariate_normal(
                cov=self.model.covariances_[k_cluster],
                mean=self.model.means_[k_cluster],
            ).logpdf(self.candidates_scaled)

            # For selecting a point from this cluster we only consider points that were
            # assigned to the current cluster by the model, hence set the density of
            # others to 0
            density[predicted_clusters != k_cluster] = 0.0

            selection.append(np.argmax(density).item())
        return selection
